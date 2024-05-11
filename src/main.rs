use std::path::Path;

use clap::Parser;
use draft::*;

fn get_base_import_target(context: &Context, target: NodeId) -> Option<(NodeId, String)> {
    let target_node = &context.nodes[target];
    match target_node {
        Node::ImportAlias { target, .. } => get_base_import_target(context, *target),
        Node::StaticMemberAccess { value, .. } => get_base_import_target(context, *value),
        Node::ImportPath { path, .. } => {
            let value = context.string_interner.resolve(path.0).unwrap();
            Some((target, value.replace("\"", "").to_string()))
        }
        _ => None,
    }
}

fn run(context: &mut Context) -> EmptyDraftResult {
    let input = context.args.input.clone();
    println!("parse_file: {}", input);
    context.parse_file(&input)?;

    let mut new_imports = Vec::new();
    std::mem::swap(&mut context.imports, &mut new_imports);

    for import in new_imports {
        let import_file_relative = context.ranges[import].source_path;
        let import_file_relative = Path::new(&import_file_relative);
        let import_file_relative = import_file_relative.parent().unwrap();

        let Node::Import { targets } = &context.nodes[import] else {
            unreachable!()
        };

        for target in targets.clone().borrow().iter() {
            if let Some((import_target_id, import_target_path)) =
                get_base_import_target(&context, *target)
            {
                let joined_path = format!(
                    "{}.dr",
                    import_file_relative
                        .join(import_target_path.clone())
                        .to_str()
                        .unwrap()
                );
                println!("parse_file: {}", joined_path);
                let module_id = context.parse_file_as_module(&joined_path)?;

                let Node::ImportPath { path, .. } = context.nodes[import_target_id] else {
                    unreachable!()
                };
                context.nodes[import_target_id] = Node::ImportPath {
                    path,
                    resolved: Some(module_id),
                };
            }
        }
    }

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
    let mut context = Context::new(args);

    match run(&mut context) {
        Ok(_) => {}
        Err(e) => context.report_error(e),
    }
}
