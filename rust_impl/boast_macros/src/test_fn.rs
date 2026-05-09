use syn::{ItemFn, Meta, parse::Parse};

pub enum TestFnArgAttr {
    Src(syn::Ident),
}
impl TestFnArgAttr {
    fn from_attr(attr: &syn::Attribute) -> syn::Result<Self> {
        fn get_str_arg(attr: &Meta) -> syn::Result<syn::Ident> {
            match attr {
                Meta::NameValue(nv) => {
                    match &nv.value {
                        // This handles: #[src = data_src]
                        syn::Expr::Path(expr_path) => {
                            if let Some(ident) = expr_path.path.get_ident() {
                                Ok(ident.clone())
                            } else {
                                Err(syn::Error::new_spanned(
                                    expr_path,
                                    "Expected a simple identifier",
                                ))
                            }
                        }
                        // This handles the old: #[src = "data_src"]
                        syn::Expr::Lit(syn::ExprLit {
                            lit: syn::Lit::Str(lit_str),
                            ..
                        }) => lit_str
                            .parse::<syn::Ident>()
                            .map_err(|_| syn::Error::new_spanned(lit_str, "Invalid identifier")),
                        _ => Err(syn::Error::new_spanned(
                            &nv.value,
                            "Expected an identifier or a string literal",
                        )),
                    }
                }
                _ => Err(syn::Error::new_spanned(attr, "Expected a name-value pair")),
            }
        }

        if attr.path().is_ident("src") {
            Ok(TestFnArgAttr::Src(get_str_arg(&attr.meta)?))
        } else {
            Err(syn::Error::new_spanned(attr, "Unknown attribute"))
        }
    }
}

pub struct TestFn {
    pub name: syn::Ident,
    pub vis: syn::Visibility,
    pub block: syn::Block,
    pub docs: Vec<syn::Attribute>,

    pub slice_type: syn::Type,
    pub slice_src_fn: syn::Ident,
}
impl Parse for TestFn {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let input_fn = input.parse::<ItemFn>()?;
        let fn_name = input_fn.sig.ident.clone();
        let fn_vis = input_fn.vis.clone();
        let fn_block = input_fn.block.clone();

        // save the docs to re-attach to the generated test function
        let docs: Vec<_> = input_fn
            .attrs
            .iter()
            .filter(|attr| attr.path().is_ident("doc"))
            .cloned()
            .collect();

        // Ensure the function takes in a single parameter - a slice of any type
        // That parameter also needs a #[src=fn_name] attribute to specify the source function for the test
        let fn_args = input_fn.sig.inputs;
        match fn_args.first() {
            Some(param) if fn_args.len() == 1 => {
                let slice_type = match param {
                    syn::FnArg::Typed(pat_type) => match &*pat_type.ty {
                        // Handle the reference layer: &[f64]
                        syn::Type::Reference(ty_ref) => match &*ty_ref.elem {
                            // Handle the slice layer: [f64]
                            syn::Type::Slice(slice) => *slice.elem.clone(),
                            _ => {
                                return Err(syn::Error::new_spanned(
                                    &ty_ref.elem,
                                    "Expected a slice [T]",
                                ));
                            }
                        },
                        _ => {
                            return Err(syn::Error::new_spanned(
                                &pat_type.ty,
                                "Parameter must be a reference to a slice (&[T])",
                            ));
                        }
                    },
                    _ => return Err(syn::Error::new_spanned(param, "Invalid function argument")),
                };

                // Extract the source function name from the #[src=fn_name] attribute on the parameter
                let attributes = match param {
                    syn::FnArg::Typed(pat_type) => &pat_type.attrs,
                    _ => unreachable!(),
                };
                let mut src_fn_name = None;
                for attr in attributes {
                    let attr = TestFnArgAttr::from_attr(attr)?;
                    match attr {
                        TestFnArgAttr::Src(src) => src_fn_name = Some(src),
                    }
                }

                let slice_src_fn = src_fn_name.ok_or_else(|| {
                    syn::Error::new_spanned(
                        param,
                        "Test function parameter must have a #[src=\"...\"] attribute",
                    )
                })?;

                Ok(TestFn {
                    name: fn_name,
                    vis: fn_vis,
                    block: *fn_block,
                    docs,
                    slice_type,
                    slice_src_fn,
                })
            }

            _ => Err(syn::Error::new_spanned(
                fn_args,
                "Test functions must take in exactly one parameter",
            )),
        }
    }
}
