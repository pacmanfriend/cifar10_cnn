use cifar10_cnn::api;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let port = parse_port(std::env::args().skip(1))?;
    api::serve(port).await
}

fn parse_port(mut args: impl Iterator<Item = String>) -> Result<u16, Box<dyn std::error::Error>> {
    let mut port = 8080u16;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--port" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--port requires a number".to_string())?;
                port = value.parse()?;
            }
            _ => return Err(format!("unknown argument '{arg}'").into()),
        }
    }
    Ok(port)
}
