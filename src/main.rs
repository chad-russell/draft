use cpi::*;

fn main() -> Result<(), CompileError> {
    let mut context = Context::new();

    // context.debug_tokens::<RopeySource>("foo.sm")?;

    context.ropey_parse_file("foo.sm")?;
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
