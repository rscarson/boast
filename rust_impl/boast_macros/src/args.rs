use syn::{
    Expr, Ident, Lit, Token,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
};

struct ArgEntry {
    key: Ident,
    _eq: Token![=],
    value: Expr, // Use Expr to allow literals, paths, etc.
}
impl Parse for ArgEntry {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        Ok(ArgEntry {
            key: input.parse()?,
            _eq: input.parse()?,
            value: input.parse()?,
        })
    }
}

struct MacroArgs {
    args: Punctuated<ArgEntry, Token![,]>,
}
impl Parse for MacroArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        Ok(MacroArgs {
            args: input.parse_terminated(ArgEntry::parse, Token![,])?,
        })
    }
}

pub struct TestArgs {
    pub confidence: f64,
    pub outlier_rate: f64,
    pub pass_ratio: Option<f64>,
    pub timeout: Option<u64>,
}
impl Parse for TestArgs {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        fn get_float_arg(value: &Expr) -> syn::Result<f64> {
            if let Expr::Lit(expr_lit) = value
                && let Lit::Float(lit_float) = &expr_lit.lit
            {
                return lit_float.base10_parse::<f64>();
            }
            Err(syn::Error::new_spanned(value, "Expected a float literal"))
        }
        fn get_int_arg(value: &Expr) -> syn::Result<u64> {
            if let Expr::Lit(expr_lit) = value
                && let Lit::Int(lit_int) = &expr_lit.lit
            {
                return lit_int.base10_parse::<u64>();
            }
            Err(syn::Error::new_spanned(
                value,
                "Expected an integer literal",
            ))
        }

        let mut confidence = None;
        let mut outlier_rate = None;
        let mut pass_ratio = None;
        let mut timeout = None;

        let macro_args = MacroArgs::parse(input)?;
        for arg in macro_args.args {
            let key = arg.key.to_string();
            match key.as_str() {
                "confidence" | "q" => confidence = Some(get_float_arg(&arg.value)?),
                "outlier_rate" | "p" => outlier_rate = Some(get_float_arg(&arg.value)?),
                "pass_ratio" => pass_ratio = Some(get_float_arg(&arg.value)?),
                "timeout" => timeout = Some(get_int_arg(&arg.value)?),
                _ => return Err(syn::Error::new_spanned(arg.key, "Unknown argument key")),
            }
        }

        let confidence = confidence.ok_or_else(|| {
            syn::Error::new(input.span(), "Missing required argument: confidence (or q)")
        })?;
        let outlier_rate = outlier_rate.ok_or_else(|| {
            syn::Error::new(
                input.span(),
                "Missing required argument: outlier_rate (or p)",
            )
        })?;

        Ok(TestArgs {
            confidence,
            outlier_rate,
            pass_ratio,
            timeout,
        })
    }
}
