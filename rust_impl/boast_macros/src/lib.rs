use proc_macro::TokenStream;
use quote::quote;
use syn::parse_macro_input;

mod args;
use args::TestArgs;

mod test_fn;
use test_fn::TestFn;

#[proc_macro_attribute]
pub fn test(args: TokenStream, input: TokenStream) -> TokenStream {
    let args = parse_macro_input!(args as TestArgs);
    let test_fn = parse_macro_input!(input as TestFn);

    // Now we can generate the test function body
    // We will call the source function to get the data
    // construct a BOAST runner with the specified arguments
    // and use the supplied test function body as the test to run on each dataset

    // for the optional arguments, we will use the builder pattern to set them if they are specified
    let confidence = args.confidence;
    let outlier_rate = args.outlier_rate;
    let mut options = quote! {
        ::boast::Options::new(#confidence, #outlier_rate)
    };
    if let Some(pass_ratio) = args.pass_ratio {
        options = quote! {
            #options.with_pass_ratio(#pass_ratio)
        };
    }
    if let Some(timeout) = args.timeout {
        options = quote! {
            #options.with_timeout(::std::time::Duration::from_secs(#timeout))
        };
    }

    let docs = test_fn.docs;
    let fn_vis = test_fn.vis;
    let fn_name = test_fn.name;
    let fn_block = test_fn.block;
    let slice_type = test_fn.slice_type;
    let src_fn_name = test_fn.slice_src_fn;

    let expanded: proc_macro2::TokenStream = quote! {
        #[test]
        #(#docs)*
        #fn_vis fn #fn_name() {
            fn _____test_fn(data: &[#slice_type]) {
                #fn_block
            }

            let data: ::boast::DataSource<_> = #src_fn_name();
            let options = #options;
            let runner = ::boast::Runner::new(options, data, _____test_fn);
            runner.run();
        }
    };

    TokenStream::from(expanded)
}
