use clap::Parser as ClapParser;

#[derive(ClapParser, Debug, Default)]
#[command(version, about)]
pub struct Args {
    pub input: String,
    
    #[arg(long)]
    pub dump_ir: bool,

    #[arg(long)]
    pub dump_tokens: bool,

    #[arg(long)]
    pub print_type_matches: bool,
}
