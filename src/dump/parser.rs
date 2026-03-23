use pom::parser::{self, *};
use super::*;

// pom::Parser is aliased to 'static; use pom::parser::Parser<'a,...> directly
type P<'a, O> = parser::Parser<'a, u8, O>;

pub fn parse_object_line_pom(line: &str) -> Result<(ObjId, DumpedObject), String> {
    object_line().parse(line.as_bytes()).map_err(|e| format!("pom: {e}"))
}

/// Parse `@{id} {type} {fields...}` into (ObjId, DumpedObject).
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

    // TODO p_hashtable

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
    kw(b"buffer").map(|_| DumpedObject::Buffer)
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

/// Parse `"escaped string"` and return the unescaped content.
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
