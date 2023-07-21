use clap::Parser;
use draft::*;

fn run(context: &mut Context) -> Result<(), CompileError> {
    let input = context.args.input.clone();
    context.parse_file(&input)?;

    if context.args.dump_tokens {
        let mut source = SourceInfo::<StrSource>::from_file(&input);
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

#[derive(PartialEq)]
struct Foo {
    x: i64,
}

fn main() {
    let args = args::Args::parse();

    let mut context = Context::new(args);

    match run(&mut context) {
        Ok(_) => {}
        Err(e) => context.report_error(e),
    }
}
