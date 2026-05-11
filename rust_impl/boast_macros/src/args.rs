use syn::{
    Expr, Ident, Token,
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
    pub confidence: Expr,
    pub outlier_rate: Expr,
    pub pass_ratio: Option<Expr>,
    pub timeout: Option<Expr>,
}
impl Parse for TestArgs {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let mut confidence = None;
        let mut outlier_rate = None;
        let mut pass_ratio = None;
        let mut timeout = None;

        let macro_args = MacroArgs::parse(input)?;
        for arg in macro_args.args {
            let key = arg.key.to_string();
            match key.as_str() {
                "confidence" | "q" => confidence = Some(arg.value),
                "outlier_rate" | "p" => outlier_rate = Some(arg.value),
                "pass_ratio" => pass_ratio = Some(arg.value),
                "timeout" => timeout = Some(arg.value),
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
