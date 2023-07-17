use draft::*;

fn run(context: &mut Context) -> Result<(), CompileError> {
    // let mut source = SourceInfo::<RopeySource>::ropey_from_file("foo.dr");
    // context.debug_tokens::<RopeySource>(&mut source)?;

    context.parse_file("foo.dr")?;
    context.prepare()?;
    if !context.errors.is_empty() {
        dbg!(&context.errors);
        return Ok(());
    }
    context.call_fn("main")?;

    // context.clear();
    // context.parse_str("fn main() i64 { return 3; }")?;
    // context.prepare()?;
    // if !context.errors.is_empty() {
    //     dbg!(&context.errors);
    //     return Ok(());
    // }
    // context.call_fn("main")?;

    Ok(())
}

fn main() {
    let mut context = Context::new();

    match run(&mut context) {
        Ok(_) => {}
        Err(e) => context.report_error(e),
    }
}
