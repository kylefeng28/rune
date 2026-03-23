///! Heap dump module for rune.
///! After a successful bootstrap in which elisp code is converted to bytecode,
///! the VM heap memory can be serialized to a rune.pdmp file, and
///! subsequent runs can avoid bootstrapping and just load rune.pdmp instead.
///!
///! The idea is based on emacs' portable dumper (pdump) which replaced the older unexec.
///! However, this pdmp format is a somewhat human-readable text format and makes no attempt to
///! have any compatibility with unexec or pdump.
///! https://lwn.net/Articles/707619/
///! https://github.com/emacs-mirror/emacs/blob/master/src/pdumper.h

use crate::core::{
    env::{Env, INTERNED_SYMBOLS},
    gc::{Context, Rt},
    object::{NIL, Object, ObjectType},
};
use std::collections::HashMap;
use std::fmt::{self, Display, Write};
use std::path::Path;

/// Unique id for each object pointer, used to handle cycles and sharing.
type ObjId = u32;

/// A serialized representation of a single GC object, decoupled from live
/// pointers. Children are referenced by ObjId rather than raw addresses.
#[derive(Debug, PartialEq)]
enum DumpedObject {
    Nil,
    Int(i64),
    Float(f64),
    String(std::string::String),
    ByteString(Vec<u8>),
    Symbol { name: std::string::String, interned: bool },
    Cons { car: ObjId, cdr: ObjId },
    Vec(Vec<ObjId>),
    ByteFn { args: u64, depth: usize, codes: Vec<u8>, consts: Vec<ObjId> },
    /// SubrFn contains a Rust fn pointer which can't be serialized.
    /// We store only the name during serialization; the loader will
    /// recover the function pointer by name lookup.
    Subr(std::string::String),
    Record(Vec<ObjId>),
    HashTable,
    Buffer,
    CharTable,
    BigInt(std::string::String),
    ChannelSender,
    ChannelReceiver,
}

/// An entry in the symbol table section: name -> (symbol object id, optional function object id)
#[derive(Debug, PartialEq)]
struct DumpedSymbol {
    name: std::string::String,
    sym_id: ObjId,
    func_id: Option<ObjId>,
}

/// An entry in the env section: a variable binding from symbol to value.
#[derive(Debug, PartialEq)]
struct DumpedBinding {
    sym_id: ObjId,
    val_id: ObjId,
}

#[derive(Debug, PartialEq)]
struct DumpState {
    /// Maps live pointer address -> assigned object id (handles cycles + dedup)
    seen: HashMap<usize, ObjId>,
    objects: Vec<DumpedObject>,
    symbols: Vec<DumpedSymbol>,
    env: Vec<DumpedBinding>,
}

impl Display for DumpedObject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Nil => write!(f, "nil"),
            Self::Int(x) => write!(f, "int {x}"),
            Self::Float(x) => write!(f, "float {x}"),
            Self::String(s) => write!(f, "string \"{}\"", escape_str(s)),
            Self::ByteString(bytes) => {
                write!(f, "bytestring #")?;
                for b in bytes {
                    write!(f, "{b:02x}")?;
                }
                Ok(())
            }
            Self::Symbol { name, interned } => {
                write!(f, "symbol \"{}\" interned={interned}", escape_str(name))
            }
            Self::Cons { car, cdr } => write!(f, "cons car=@{car} cdr=@{cdr}"),
            Self::Vec(elems) => {
                write!(f, "vec [")?;
                fmt_id_list(f, elems)?;
                write!(f, "]")
            }
            Self::ByteFn { args, depth, codes, consts } => {
                write!(f, "bytefn args={args} depth={depth} codes=#")?;
                for b in codes {
                    write!(f, "{b:02x}")?;
                }
                write!(f, " consts=[")?;
                fmt_id_list(f, consts)?;
                write!(f, "]")
            }
            Self::Subr(name) => write!(f, "subr \"{}\"", escape_str(name)),
            Self::Record(elems) => {
                write!(f, "record [")?;
                fmt_id_list(f, elems)?;
                write!(f, "]")
            }
            Self::HashTable => write!(f, "hashtable <opaque>"),
            Self::Buffer => write!(f, "buffer <opaque>"),
            Self::CharTable => write!(f, "chartable <opaque>"),
            Self::BigInt(s) => write!(f, "bigint {s}"),
            Self::ChannelSender => write!(f, "channel-sender <opaque>"),
            Self::ChannelReceiver => write!(f, "channel-receiver <opaque>"),
        }
    }
}

fn fmt_id_list(f: &mut fmt::Formatter<'_>, ids: &[ObjId]) -> fmt::Result {
    for (i, id) in ids.iter().enumerate() {
        if i > 0 {
            write!(f, " ")?;
        }
        write!(f, "@{id}")?;
    }
    Ok(())
}

impl DumpState {
    fn new() -> Self {
        Self {
            seen: HashMap::new(),
            objects: Vec::new(),
            symbols: Vec::new(),
            env: Vec::new(),
        }
    }

    /// Serialize an object and return its id. If already visited (cycle or
    /// shared reference), returns the cached id immediately.
    fn dump_object(&mut self, obj: Object) -> ObjId {
        let addr = obj.into_raw().addr();

        // nil is special: always id 0
        if obj.ptr_eq(NIL) {
            if !self.seen.contains_key(&addr) {
                let id = self.alloc_id(DumpedObject::Nil);
                self.seen.insert(addr, id);
            }
            return self.seen[&addr];
        }

        // If already visited -> return cached id (handles cycles + sharing)
        if let Some(&id) = self.seen.get(&addr) {
            return id;
        }

        // Pre-insert before recursing into children to break cycles.
        // If a child points back to this object, dump_object will find
        // it in `seen` and return the id without infinite recursion.
        let id = self.alloc_id(DumpedObject::Nil);
        self.seen.insert(addr, id);

        let dumped = match obj.untag() {
            ObjectType::Int(x) => DumpedObject::Int(x),
            ObjectType::Float(x) => DumpedObject::Float(**x),
            ObjectType::String(s) => DumpedObject::String(s.inner().to_owned()),
            ObjectType::ByteString(s) => DumpedObject::ByteString(s.to_vec()),
            ObjectType::Symbol(sym) => DumpedObject::Symbol {
                name: sym.name().to_owned(),
                interned: sym.interned(),
            },
            ObjectType::Cons(cons) => {
                let car = self.dump_object(cons.car());
                let cdr = self.dump_object(cons.cdr());
                DumpedObject::Cons { car, cdr }
            }
            ObjectType::Vec(vec) => {
                let elems = vec.iter().map(|cell| self.dump_object(cell.get())).collect();
                DumpedObject::Vec(elems)
            }
            ObjectType::ByteFn(bf) => {
                // Bytecode is trivially serializable; the constants
                // vector contains arbitrary Objects so we recurse into each.
                let consts = bf.consts().iter().map(|c| self.dump_object(*c)).collect();
                DumpedObject::ByteFn {
                    args: bf.args.into_arg_spec(),
                    depth: bf.depth,
                    codes: bf.codes().to_vec(),
                    consts,
                }
            }
            ObjectType::SubrFn(subr) => DumpedObject::Subr(subr.name.to_owned()),
            ObjectType::HashTable(_) => DumpedObject::HashTable,
            ObjectType::Record(rec) => {
                let elems = rec.iter().map(|cell| self.dump_object(cell.get())).collect();
                DumpedObject::Record(elems)
            }
            ObjectType::Buffer(_) => DumpedObject::Buffer,
            ObjectType::CharTable(_) => DumpedObject::CharTable,
            ObjectType::BigInt(n) => DumpedObject::BigInt(n.to_string()),
            ObjectType::ChannelSender(_) => DumpedObject::ChannelSender,
            ObjectType::ChannelReceiver(_) => DumpedObject::ChannelReceiver,
        };

        // Overwrite the Nil placeholder with the real object
        self.objects[id as usize] = dumped;
        id
    }

    fn alloc_id(&mut self, placeholder: DumpedObject) -> ObjId {
        let id = self.objects.len() as ObjId;
        self.objects.push(placeholder);
        id
    }

    /// Walk the global symbol table and serialize each symbol + its function cell.
    fn dump_symbols(&mut self, cx: &Context) {
        let map = INTERNED_SYMBOLS.lock().unwrap();
        for (name, sym) in map.iter() {
            let sym_obj: Object = sym.into();
            let sym_id = self.dump_object(sym_obj);
            let func_id = if sym.has_func() {
                let func = sym.func(cx).unwrap();
                Some(self.dump_object(func.as_obj()))
            } else {
                None
            };
            self.symbols.push(DumpedSymbol {
                name: name.to_owned(),
                sym_id,
                func_id,
            });
        }
    }

    /// Serialize the Env variable bindings (symbol -> value mappings).
    fn dump_env(&mut self, env: &Rt<Env>, _cx: &Context) {
        for (sym_slot, val_slot) in env.vars.iter() {
            // Slot<Symbol> derefs to Symbol, Slot<Object> derefs to Object.
            // Double-deref: &Slot<T> -> &T -> T (Copy).
            let sym_obj: Object = (**sym_slot).into();
            let val: Object = **val_slot;
            let sym_id = self.dump_object(sym_obj);
            let val_id = self.dump_object(val);
            self.env.push(DumpedBinding { sym_id, val_id });
        }
    }

    fn into_output(self) -> std::string::String {
        let mut out = std::string::String::new();
        writeln!(out, ".HEADER").unwrap();
        writeln!(out, "  version 1").unwrap();
        writeln!(out, "  objects {}", self.objects.len()).unwrap();
        writeln!(out).unwrap();

        writeln!(out, ".OBJECTS").unwrap();
        for (id, obj) in self.objects.iter().enumerate() {
            writeln!(out, "  @{id} {obj}").unwrap();
        }
        writeln!(out).unwrap();

        writeln!(out, ".SYMBOLS").unwrap();
        for sym in &self.symbols {
            let escaped = escape_str(&sym.name);
            match sym.func_id {
                Some(fid) => writeln!(out, "  \"{escaped}\" -> @{} func=@{fid}", sym.sym_id),
                None => writeln!(out, "  \"{escaped}\" -> @{}", sym.sym_id),
            }
            .unwrap();
        }
        writeln!(out).unwrap();

        writeln!(out, ".ENV").unwrap();
        for b in &self.env {
            writeln!(out, "  @{} = @{}", b.sym_id, b.val_id).unwrap();
        }
        out
    }
}

fn escape_str(s: &str) -> std::string::String {
    let mut out = std::string::String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out
}

// ── Serialization ───────────────────────────────────────────────────────────

pub(crate) fn dump_to_file(
    path: &Path,
    env: &Rt<Env>,
    cx: &Context,
) -> Result<(), std::io::Error> {
    let mut state = DumpState::new();
    state.dump_symbols(cx);
    state.dump_env(env, cx);
    let output = state.into_output();
    std::fs::write(path, output)
}

/// Registry mapping SubrFn names to their static references.
/// Used by the loader to reconstruct function pointers from dump files.
/// We serialize the name since Rust fn pointer names can't be serialized,
/// and look it up here on load.
fn subr_registry() -> HashMap<&'static str, &'static crate::core::object::SubrFn> {
    use crate::core::env::sym;
    let mut map = HashMap::new();
    for subr in sym::SUBR_DEFS.iter() {
        map.insert(subr.name, *subr);
    }
    map
}

// ── Deserialization ─────────────────────────────────────────────────────────

/// Deserialized contents of a .pdmp file.
struct DumpFile {
    objects: Vec<DumpedObject>,
    symbols: Vec<DumpedSymbol>,
    env: Vec<DumpedBinding>,
}

fn parse_dump(input: &str) -> Result<DumpFile, String> {
    let mut objects: Vec<Option<DumpedObject>> = Vec::new();
    let mut symbols = Vec::new();
    let mut env = Vec::new();
    let mut section = "";

    for line in input.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('.') {
            section = match line {
                ".HEADER" => "header",
                ".OBJECTS" => "objects",
                ".SYMBOLS" => "symbols",
                ".ENV" => "env",
                _ => return Err(format!("unknown section: {line}")),
            };
            continue;
        }
        match section {
            "header" => {} // skip version/count lines
            "objects" => {
                // let (id, obj) = parse_object_line(line)?;
                let (id, obj) = parser::parse_object_line_pom(line)?;
                // Grow vec to fit, filling gaps with None
                if id as usize >= objects.len() {
                    objects.resize_with(id as usize + 1, || None);
                }
                objects[id as usize] = Some(obj);
            }
            "symbols" => symbols.push(parse_symbol_line(line)?),
            "env" => env.push(parse_binding_line(line)?),
            _ => {}
        }
    }

    let objects = objects
        .into_iter()
        .enumerate()
        .map(|(i, o)| o.ok_or_else(|| format!("missing object @{i}")))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(DumpFile { objects, symbols, env })
}

/// Parse `@{id} {type} {fields...}` into (ObjId, DumpedObject).
fn parse_object_line(line: &str) -> Result<(ObjId, DumpedObject), String> {
    let line = line.strip_prefix('@').ok_or("expected @")?;
    let (id_str, rest) = line.split_once(' ').ok_or("expected space after id")?;
    let id: ObjId = id_str.parse().map_err(|e| format!("bad id: {e}"))?;

    let obj = if rest == "nil" {
        DumpedObject::Nil
    } else if let Some(val) = rest.strip_prefix("int ") {
        DumpedObject::Int(val.parse().map_err(|e| format!("bad int: {e}"))?)
    } else if let Some(val) = rest.strip_prefix("float ") {
        DumpedObject::Float(val.parse().map_err(|e| format!("bad float: {e}"))?)
    } else if let Some(val) = rest.strip_prefix("string ") {
        DumpedObject::String(parse_quoted_string(val)?)
    } else if let Some(val) = rest.strip_prefix("bytestring #") {
        DumpedObject::ByteString(parse_hex_bytes(val)?)
    } else if let Some(val) = rest.strip_prefix("symbol ") {
        let (name, rest) = parse_quoted_string_rest(val)?;
        let interned = rest
            .strip_prefix(" interned=")
            .ok_or("expected interned=")?
            .parse()
            .map_err(|e| format!("bad bool: {e}"))?;
        DumpedObject::Symbol { name, interned }
    } else if let Some(val) = rest.strip_prefix("cons ") {
        let car = parse_field_ref(val, "car=@")?;
        let cdr_start = val.find("cdr=@").ok_or("expected cdr=")?;
        let cdr = parse_ref(&val[cdr_start + "cdr=".len()..])?;
        DumpedObject::Cons { car, cdr }
    } else if let Some(val) = rest.strip_prefix("vec [") {
        let inner = val.strip_suffix(']').ok_or("expected ]")?;
        DumpedObject::Vec(parse_ref_list(inner)?)
    } else if let Some(val) = rest.strip_prefix("bytefn ") {
        parse_bytefn(val)?
    } else if let Some(val) = rest.strip_prefix("subr ") {
        DumpedObject::Subr(parse_quoted_string(val)?)
    } else if let Some(val) = rest.strip_prefix("record [") {
        let inner = val.strip_suffix(']').ok_or("expected ]")?;
        DumpedObject::Record(parse_ref_list(inner)?)
    } else if let Some(val) = rest.strip_prefix("bigint ") {
        DumpedObject::BigInt(val.to_owned())
    } else if rest.starts_with("hashtable") {
        DumpedObject::HashTable
    } else if rest.starts_with("buffer") {
        DumpedObject::Buffer
    } else if rest.starts_with("chartable") {
        DumpedObject::CharTable
    } else if rest.starts_with("channel-sender") {
        DumpedObject::ChannelSender
    } else if rest.starts_with("channel-receiver") {
        DumpedObject::ChannelReceiver
    } else {
        return Err(format!("unknown object type: {rest}"));
    };

    Ok((id, obj))
}

fn parse_bytefn(val: &str) -> Result<DumpedObject, String> {
    // args={n} depth={n} codes=#{hex} consts=[...]
    let args_str = extract_field(val, "args=", ' ')?;
    let args: u64 = args_str.parse().map_err(|e| format!("bad args: {e}"))?;
    let depth_str = extract_field(val, "depth=", ' ')?;
    let depth: usize = depth_str.parse().map_err(|e| format!("bad depth: {e}"))?;
    let codes_start = val.find("codes=#").ok_or("expected codes=#")? + 7;
    let codes_end = val[codes_start..].find(' ').map_or(val.len(), |i| codes_start + i);
    let codes = parse_hex_bytes(&val[codes_start..codes_end])?;
    let consts_start = val.find("consts=[").ok_or("expected consts=[")? + 8;
    let consts_end = val[consts_start..].find(']').ok_or("expected ]")? + consts_start;
    let consts = parse_ref_list(&val[consts_start..consts_end])?;
    Ok(DumpedObject::ByteFn { args, depth, codes, consts })
}

/// Parse `"escaped string"` and return the unescaped content.
fn parse_quoted_string(s: &str) -> Result<std::string::String, String> {
    parse_quoted_string_rest(s).map(|(s, _)| s)
}

/// Parse `"escaped string" rest...` returning (unescaped, rest).
fn parse_quoted_string_rest(s: &str) -> Result<(std::string::String, &str), String> {
    let s = s.strip_prefix('"').ok_or("expected opening quote")?;
    let mut out = std::string::String::new();
    let mut chars = s.char_indices();
    while let Some((i, c)) = chars.next() {
        match c {
            '"' => return Ok((out, &s[i + 1..])),
            '\\' => match chars.next() {
                Some((_, 'n')) => out.push('\n'),
                Some((_, 'r')) => out.push('\r'),
                Some((_, 't')) => out.push('\t'),
                Some((_, '\\')) => out.push('\\'),
                Some((_, '"')) => out.push('"'),
                Some((_, c)) => {
                    out.push('\\');
                    out.push(c);
                }
                None => return Err("unexpected end of string".into()),
            },
            c => out.push(c),
        }
    }
    Err("unterminated string".into())
}

fn parse_hex_bytes(s: &str) -> Result<Vec<u8>, String> {
    (0..s.len())
        .step_by(2)
        .map(|i| {
            u8::from_str_radix(s.get(i..i + 2).ok_or("odd hex length")?, 16)
                .map_err(|e| format!("bad hex: {e}"))
        })
        .collect()
}

/// Parse `@{id}` returning the id.
fn parse_ref(s: &str) -> Result<ObjId, String> {
    let s = s.trim();
    let digits = s.strip_prefix('@').ok_or_else(|| format!("expected @, got: {s}"))?;
    // Take only digits (stop at space or end)
    let end = digits.find(|c: char| !c.is_ascii_digit()).unwrap_or(digits.len());
    digits[..end].parse().map_err(|e| format!("bad ref: {e}"))
}

/// Parse `prefix{id}` from the start of s.
fn parse_field_ref(s: &str, prefix: &str) -> Result<ObjId, String> {
    let start = s.find(prefix).ok_or_else(|| format!("expected {prefix}"))?;
    parse_ref(&s[start + prefix.len() - 1..]) // -1 to keep the @
}

/// Parse space-separated `@id @id ...` list.
fn parse_ref_list(s: &str) -> Result<Vec<ObjId>, String> {
    let s = s.trim();
    if s.is_empty() {
        return Ok(Vec::new());
    }
    s.split_whitespace().map(|tok| parse_ref(tok)).collect()
}

/// Extract value between `key` and `delim` (or end of string).
fn extract_field<'a>(s: &'a str, key: &str, delim: char) -> Result<&'a str, String> {
    let start = s.find(key).ok_or_else(|| format!("expected {key}"))? + key.len();
    let end = s[start..].find(delim).map_or(s.len(), |i| start + i);
    Ok(&s[start..end])
}

/// Parse `"name" -> @{id}` or `"name" -> @{id} func=@{fid}`.
fn parse_symbol_line(line: &str) -> Result<DumpedSymbol, String> {
    let (name, rest) = parse_quoted_string_rest(line)?;
    let rest = rest.strip_prefix(" -> ").ok_or("expected -> ")?;
    let sym_id = parse_ref(rest)?;
    let func_id = if let Some(fpos) = rest.find("func=@") {
        Some(parse_ref(&rest[fpos + "func=".len()..])?)
    } else {
        None
    };
    Ok(DumpedSymbol { name, sym_id, func_id })
}

/// Parse `@{sym_id} = @{val_id}`.
fn parse_binding_line(line: &str) -> Result<DumpedBinding, String> {
    let (left, right) = line.split_once(" = ").ok_or("expected = ")?;
    Ok(DumpedBinding {
        sym_id: parse_ref(left)?,
        val_id: parse_ref(right)?,
    })
}

// ── Loader ──────────────────────────────────────────────────────────────────
// Two-pass reconstruction of the GC heap from a parsed DumpFile.
//
// Pass 1: Allocate a placeholder GC object for every DumpedObject entry,
//         building an id->Object lookup table.
//         Leaf types (int, float, string, symbol, subr) are fully constructed here.
//         Compound types (cons, vec, bytefn) get placeholders with nil/empty contents.
//         SubrFn is a special case: only the name of the function pointer is
//         serialized, so we look up the static reference in `sym::SUBR_DEFS`
//
// Pass 2: Fix up pointer fields in compound objects: patch cons car/cdr,
//         vec elements, and bytefn constants with the real Objects from the
//         lookup table.
//
// After both passes, we rebuild the symbol function cells and env var bindings and inject
// it into the runtime.

use crate::core::{
    cons::Cons,
    env::intern,
    object::{FnArgs, IntoObject, Symbol},
};

pub(crate) fn load_dump(
    path: &Path,
    env: &mut Rt<Env>,
    cx: &mut Context,
) -> Result<(), String> {
    let input = std::fs::read_to_string(path).map_err(|e| format!("read error: {e}"))?;
    let dump = parse_dump(&input)?;
    let subrs = subr_registry();

    // Pass 1: allocate placeholder GC objects and build a id->Object lookup table
    let mut table: Vec<Object> = Vec::with_capacity(dump.objects.len());

    for obj in &dump.objects {
        let live: Object = match obj {
            DumpedObject::Nil => NIL,
            DumpedObject::Int(x) => cx.add(*x),
            DumpedObject::Float(x) => cx.add(*x),
            DumpedObject::String(s) => cx.add(s.as_str()),
            DumpedObject::ByteString(b) => cx.add(b.clone()),
            DumpedObject::Symbol { name, interned } => {
                if *interned {
                    // Interned symbols already exist in the global table;
                    // look them up rather than creating duplicates.
                    let sym = intern(name, cx);
                    let obj: Object = sym.into();
                    obj
                } else {
                    let sym = Symbol::new_uninterned(name, cx);
                    let obj: Object = sym.into();
                    obj
                }
            }
            // Cons: allocate with nil/nil, will be patched in pass 2
            DumpedObject::Cons { .. } => {
                let cons = Cons::new(NIL, NIL, cx);
                let obj: Object = cons.into();
                obj
            }
            // Vec: allocate with correct length, filled with nil, patched in pass 2
            DumpedObject::Vec(elems) => {
                let nils: Vec<Object> = vec![NIL; elems.len()];
                cx.add(nils)
            }
            DumpedObject::ByteFn { args, depth, codes, consts } => {
                // Allocate constants vector with nil placeholders, patched in pass 2
                let nils: Vec<Object> = vec![NIL; consts.len()];
                let const_vec: crate::core::object::Gc<&crate::core::object::LispVec> =
                    nils.into_obj(cx);
                let fn_args = FnArgs::from_arg_spec(*args as i64)
                    .map_err(|e| format!("bad arg spec: {e}"))?;
                let bytefn = unsafe {
                    crate::core::object::ByteFn::make(codes, const_vec.untag(), fn_args, *depth)
                };
                let obj: Object = bytefn.into_obj(cx).into();
                obj
            }
            DumpedObject::Subr(name) => {
                // Look up the Rust function pointer by name
                let subr = subrs
                    .get(name.as_str())
                    .ok_or_else(|| format!("unknown subr: {name}"))?;
                let obj: Object = (*subr).into();
                obj
            }
            DumpedObject::Record(elems) => {
                let nils: Vec<Object> = vec![NIL; elems.len()];
                let mut gvec = cx.vec_with_capacity(nils.len());
                gvec.extend_from_slice(&nils);
                let builder = crate::core::object::RecordBuilder(gvec);
                let obj: Object = builder.into_obj(cx).into();
                obj
            }
            DumpedObject::BigInt(s) => {
                let n: num_bigint::BigInt =
                    s.parse().map_err(|e| format!("bad bigint: {e}"))?;
                cx.add(n)
            }
            // Opaque types — we can't reconstruct these, use nil as placeholder
            DumpedObject::HashTable
            | DumpedObject::Buffer
            | DumpedObject::CharTable
            | DumpedObject::ChannelSender
            | DumpedObject::ChannelReceiver => NIL,
        };
        table.push(live);
    }

    // Pass 2: fix up compound object pointers
    for (id, obj) in dump.objects.iter().enumerate() {
        match obj {
            DumpedObject::Cons { car, cdr } => {
                let cons = table[id].untag();
                if let ObjectType::Cons(c) = cons {
                    // Cons cells are allocated mutable, so set_car/set_cdr work
                    c.set_car(table[*car as usize]).map_err(|e| e.to_string())?;
                    c.set_cdr(table[*cdr as usize]).map_err(|e| e.to_string())?;
                }
            }
            DumpedObject::Vec(elems) => {
                if let ObjectType::Vec(v) = table[id].untag() {
                    let cells = v.try_mut().map_err(|e| e.to_string())?;
                    for (i, elem_id) in elems.iter().enumerate() {
                        cells[i].set(table[*elem_id as usize]);
                    }
                }
            }
            DumpedObject::ByteFn { consts, .. } => {
                if let ObjectType::ByteFn(bf) = table[id].untag() {
                    // The constants vector was allocated with nil placeholders;
                    // patch each element with the real object.
                    let cells = bf.consts_mut().map_err(|e| e.to_string())?;
                    for (i, c_id) in consts.iter().enumerate() {
                        cells[i].set(table[*c_id as usize]);
                    }
                }
            }
            DumpedObject::Record(elems) => {
                if let ObjectType::Record(r) = table[id].untag() {
                    let cells = r.try_mut().map_err(|e| e.to_string())?;
                    for (i, elem_id) in elems.iter().enumerate() {
                        cells[i].set(table[*elem_id as usize]);
                    }
                }
            }
            _ => {} // leaf types are already fully constructed
        }
    }

    // Rebuild symbol function cells
    // For symbols that had a function binding at dump time, re-bind it.
    // We go through INTERNED_SYMBOLS.set_func which clones the function
    // into the global block and marks it immutable.
    {
        let map = INTERNED_SYMBOLS.lock().unwrap();
        for sym_entry in &dump.symbols {
            if let Some(func_id) = sym_entry.func_id {
                let func_obj = table[func_id as usize];
                // Only re-bind non-subr functions — subrs are already bound
                // by init_symbols(). Elisp-defined functions (ByteFn, Cons
                // closures) from bootstrap are what we need to restore.
                if let ObjectType::SubrFn(_) = func_obj.untag() {
                    continue;
                }
                if let Some(sym) = map.get(&sym_entry.name) {
                    let func: crate::core::object::Function = unsafe {
                        crate::core::object::Gc::from_raw(func_obj.into_raw())
                    };
                    // set_func clones into the global block and marks immutable
                    map.set_func(sym, func).map_err(|e| e.to_string())?;
                }
            }
        }
    }

    // Rebuild env variable bindings
    for binding in &dump.env {
        let sym_obj = table[binding.sym_id as usize];
        let val_obj = table[binding.val_id as usize];
        if let ObjectType::Symbol(sym) = sym_obj.untag() {
            env.vars.insert(sym, val_obj);
        }
    }

    Ok(())
}

mod parser {
    use pom::parser::{self, *};
    use super::*;

    // pom::Parser is aliased to 'static; use pom::parser::Parser<'a,...> directly
    type P<'a, O> = parser::Parser<'a, u8, O>;

    pub fn parse_object_line_pom(line: &str) -> Result<(ObjId, DumpedObject), String> {
        object_line().parse(line.as_bytes()).map_err(|e| format!("pom: {e}"))
    }

    fn object_line<'a>() -> P<'a, (ObjId, DumpedObject)> {
        let p_nil = kw(b"nil").map(|_| DumpedObject::Nil);

        let p_int = kw(b"int ") * (signed_digits())
            .convert(String::from_utf8)
            .convert(|s| s.parse::<i64>())
            .map(DumpedObject::Int);

        let p_float = kw(b"float ") * is_a(|b: u8| !b.is_ascii_whitespace()).repeat(1..)
            .convert(String::from_utf8)
            .convert(|s: String| s.parse::<f64>())
            .map(DumpedObject::Float);

        let p_string = (kw(b"string ") * quoted_string()).map(DumpedObject::String);

        let p_bytestring = (kw(b"bytestring ") * hex_bytes()).map(DumpedObject::ByteString);

        let p_symbol = kw(b"symbol ") * (quoted_string() - kw(b" interned=") + (
                kw(b"true").map(|_| true) | kw(b"false").map(|_| false)
        )).map(|(name, interned)| DumpedObject::Symbol { name, interned });

        let p_cons = kw(b"cons car=") * (obj_ref() - kw(b" cdr=") + obj_ref())
            .map(|(car, cdr)| DumpedObject::Cons { car, cdr });

        let p_vec = (kw(b"vec ") * ref_list()).map(DumpedObject::Vec);
        let p_subr = (kw(b"subr ") * quoted_string()).map(DumpedObject::Subr);
        let p_record = (kw(b"record ") * ref_list()).map(DumpedObject::Record);

        let p_bigint = (kw(b"bigint ") * is_a(|b: u8| !b.is_ascii_whitespace()).repeat(1..))
            .convert(String::from_utf8)
            .map(DumpedObject::BigInt);

        obj_ref() - sp() + (
            p_nil | p_int | p_float | p_string | p_bytestring
            | p_symbol | p_cons | p_vec
            | p_bytefn() | p_subr
            | p_record | p_bigint
            | p_opaque()
        )
    }

    fn p_bytefn<'a>() -> P<'a, DumpedObject> {
        let args = kw(b"bytefn args=") * parse_digits::<u64>();
        let depth = kw(b" depth=") * parse_digits::<usize>();
        let codes = kw(b" codes=") * hex_bytes();
        let consts = kw(b" consts=") * ref_list();
        (args + depth + codes + consts)
            .map(|(((args, depth), codes), consts)| {
                DumpedObject::ByteFn { args, depth, codes, consts }
            })
    }

    fn p_opaque<'a>() -> P<'a, DumpedObject> {
        kw(b"hashtable").map(|_| DumpedObject::HashTable)
        | kw(b"buffer").map(|_| DumpedObject::Buffer)
        | kw(b"chartable").map(|_| DumpedObject::CharTable)
        | kw(b"channel-sender").map(|_| DumpedObject::ChannelSender)
        | kw(b"channel-receiver").map(|_| DumpedObject::ChannelReceiver)
    }

    // Helpers
    fn kw<'a>(t: &'static [u8]) -> P<'a, ()> {
        seq(t).discard()
    }

    fn sp<'a>() -> P<'a, ()> {
        sym(b' ').repeat(1..).discard()
    }

    fn digits<'a>() -> P<'a, Vec<u8>> {
        is_a(|b: u8| b.is_ascii_digit()).repeat(1..)

    }

    fn signed_digits<'a>() -> P<'a, Vec<u8>> {
        (sym(b'-').opt() + digits()).map(|(sign, digits)| {
            if let Some(sign) = sign {
                let mut d = digits.clone();
                d.splice(0..0, [sign]);
                d
            } else { digits }
        })
    }

    fn parse_digits<'a, T: std::str::FromStr + 'a>() -> P<'a, T>
    where T::Err: std::fmt::Debug
    {
        digits().convert(String::from_utf8).convert(|s: String| s.parse::<T>())
    }

    fn obj_ref<'a>() -> P<'a, ObjId> {
        sym(b'@') * parse_digits::<ObjId>()
    }

    fn ref_list<'a>() -> P<'a, Vec<ObjId>> {
        sym(b'[') * list(obj_ref(), sp()).opt().map(|o: Option<Vec<ObjId>>| o.unwrap_or_default()) - sym(b']')
    }

    fn quoted_string<'a>() -> P<'a, String> {
        let escape = sym(b'\\') * (
            sym(b'n').map(|_| b'\n')
            | sym(b'r').map(|_| b'\r')
            | sym(b't').map(|_| b'\t')
            | sym(b'\\').map(|_| b'\\')
            | sym(b'"').map(|_| b'"')
        );
        let char_or_esc = none_of(b"\\\"") | escape;
        (sym(b'"') * char_or_esc.repeat(0..) - sym(b'"'))
            .convert(String::from_utf8)
    }

    fn hex_bytes<'a>() -> P<'a, Vec<u8>> {
        (sym(b'#') * is_a(|b: u8| b.is_ascii_hexdigit()).repeat(0..))
            .convert(String::from_utf8)
            .map(|s| {
                (0..s.len()).step_by(2)
                    .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
                    .collect()
            })
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_pom_nil() {
            assert_eq!(
                parse_object_line_pom("@0 nil").unwrap(),
                (0, DumpedObject::Nil)
            );
        }

        #[test]
        fn test_pom_int() {
            assert_eq!(
                parse_object_line_pom("@5 int -42").unwrap(),
                (5, DumpedObject::Int(-42))
            );
        }

        #[test]
        fn test_pom_cons() {
            assert_eq!(
                parse_object_line_pom("@8 cons car=@4 cdr=@5").unwrap(),
                (8, DumpedObject::Cons { car: 4, cdr: 5 })
            );
        }

        #[test]
        fn test_pom_string() {
            assert_eq!(
                parse_object_line_pom(r#"@6 string "hello world""#).unwrap(),
                (6, DumpedObject::String("hello world".into()))
            );
        }

        #[test]
        fn test_pom_bytefn() {
            assert_eq!(
                parse_object_line_pom("@10 bytefn args=513 depth=3 codes=#c70817 consts=[@6 @4]").unwrap(),
                (10, DumpedObject::ByteFn {
                    args: 513, depth: 3,
                    codes: vec![0xc7, 0x08, 0x17],
                    consts: vec![6, 4],
                })
            );
        }

        #[test]
        fn test_pom_vec_empty() {
            assert_eq!(
                parse_object_line_pom("@9 vec []").unwrap(),
                (9, DumpedObject::Vec(vec![]))
            );
        }

        #[test]
        fn test_pom_symbol() {
            assert_eq!(
                parse_object_line_pom(r#"@2 symbol "bare-symbol" interned=true"#).unwrap(),
                (2, DumpedObject::Symbol { name: "bare-symbol".into(), interned: true })
            );
        }
    }
}
