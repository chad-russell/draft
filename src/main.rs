use clap::Parser;
use draft::*;

fn run(context: &mut Context) -> Result<(), CompileError> {
    context.parse_file("foo.dr")?;

    if context.args.dump_tokens {
        let mut source = SourceInfo::<StrSource>::from_file("foo.dr");
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

    let mut context = Context::new(args);

    match run(&mut context) {
        Ok(_) => {}
        Err(e) => context.report_error(e),
    }
}
