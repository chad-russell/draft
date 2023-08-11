#![feature(allocator_api)]

use bumpalo::Bump;
use clap::Parser;
use draft::*;

use tracing::instrument;
use tracing_subscriber::layer::SubscriberExt;

#[instrument(skip_all)]
fn run(context: &mut Context) -> DraftResult<()> {
    let input = context.args.input.clone();
    context.parse_file(&input)?;

    if context.args.dump_tokens {
        let mut source = context.make_ropey_source_info_from_file(&input);
        context.debug_tokens(&mut source)?;
        return Ok(());
    }

    context.prepare()?;

    if !context.errors.is_empty() {
        dbg!(&context.errors);
        return Ok(());
    }

    context.call_fn("main")?;

    Ok(())
}

fn main() {
    let args = args::Args::parse();
    let bump = Box::leak(Box::new(Bump::new()));
    let mut context = Context::new(args, bump);

    if context.args.tracing {
        tracing::subscriber::set_global_default(
            tracing_subscriber::registry().with(tracing_tracy::TracyLayer::new()),
        )
        .expect("Failed to set tracing_tracy subscriber");

        // loop {
        run(&mut context).unwrap();
        // context.reset();
        // }
    } else {
        match run(&mut context) {
            Ok(_) => {}
            Err(e) => context.report_error(e),
        }
    }
}
