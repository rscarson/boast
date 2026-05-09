use std::io::Write;

enum DataSource {
    JsonFile(String),
    Command(String),
}
impl DataSource {
    pub fn load_data(&self) -> Result<boast::DataSource<String>, std::io::Error> {
        match self {
            Self::JsonFile(path) => {
                //
                // Load the whole set ahead of time - bad but workable for now. I can add streaming support later

                let file = std::fs::File::open(path)?;
                let reader = std::io::BufReader::new(file);
                let data: Vec<Vec<serde_json::Value>> = serde_json::from_reader(reader)?;

                // Convert the data to a Vec<Vec<String>>, which is the expected format for the test function
                let data: Vec<Vec<String>> = data
                    .into_iter()
                    .map(|inner| {
                        inner
                            .into_iter()
                            .map(|value| value.to_string())
                            .collect::<Vec<String>>()
                    })
                    .collect();

                if data.is_empty() {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "JSON file is empty",
                    ));
                }

                Ok(boast::DataSource::iterable(
                    data.first().unwrap().len(),
                    data.into_iter(),
                ))
            }

            Self::Command(cmd) => {
                //
                // The data source will just run the command and parse the result as a set of linebreak-delimited strings
                // Again this is pretty bad but it allows for maximum flexibility in the short term. I can add streaming support later probly

                let run_cmd = |cmd: String, seed: u64| {
                    let parts = cmd.split_whitespace().collect::<Vec<_>>();
                    if parts.is_empty() {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            "Command cannot be empty",
                        ));
                    }

                    let mut output = std::process::Command::new(parts[0]);
                    output.arg(seed.to_string());
                    if parts.len() > 1 {
                        output.args(&parts[1..]);
                    }

                    let output = output.output()?;
                    if !output.status.success() {
                        return Err(std::io::Error::other(format!(
                            "Command failed with status: {}",
                            output.status
                        )));
                    }

                    let stdout = String::from_utf8_lossy(&output.stdout);
                    let data: Vec<String> = stdout.lines().map(|line| line.to_string()).collect();

                    if data.is_empty() {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            "Command output is empty",
                        ));
                    }

                    Ok(data)
                };

                // Run the command once to get the length of the data, which is needed for the DataSource
                let data = run_cmd(cmd.to_string(), 0)?;
                let len = data.len();

                let cmd = cmd.to_string();
                Ok(boast::DataSource::generated(len, move |dst, seed| {
                    let data = run_cmd(cmd.clone(), seed).expect("Failed to run command");
                    if dst.len() != data.len() {
                        panic!("Command output length does not match expected length");
                    }

                    for (dst_item, data_item) in dst.iter_mut().zip(data.into_iter()) {
                        *dst_item = data_item;
                    }
                }))
            }
        }
    }
}

struct CliArgs {
    q: f64,
    p: f64,
    pass_ratio: Option<f64>,
    timeout: Option<u64>,

    data_source: DataSource,
    test_cmd: String,
}
impl CliArgs {
    fn parse() -> Result<Self, &'static str> {
        let mut args_iter = std::env::args().skip(1);

        let mut q = None;
        let mut p = None;
        let mut pass_ratio = None;
        let mut timeout = None;
        let mut data_source = None;
        let mut test_cmd = None;

        while let Some(arg) = args_iter.next() {
            match arg.as_str() {
                "-q" | "--confidence" => {
                    let value = args_iter.next().ok_or("Missing value for confidence")?;
                    let value = value
                        .parse::<f64>()
                        .map_err(|_| "confidence should be a float from 0.0 to 1.0")?;
                    q = Some(value);
                }

                "-p" | "--outlier_rate" => {
                    let value = args_iter.next().ok_or("Missing value for outlier_rate")?;
                    let value = value
                        .parse::<f64>()
                        .map_err(|_| "outlier_rate should be a float from 0.0 to 1.0")?;
                    p = Some(value);
                }

                "--pass_ratio" => {
                    let value = args_iter.next().ok_or("Missing value for pass_ratio")?;
                    let value = value
                        .parse::<f64>()
                        .map_err(|_| "pass_ratio should be a float from 0.0 to 1.0")?;
                    pass_ratio = Some(value);
                }

                "--timeout" => {
                    let value = args_iter.next().ok_or("Missing value for timeout")?;
                    let value = value
                        .parse::<u64>()
                        .map_err(|_| "timeout should be an integer representing seconds")?;
                    timeout = Some(value);
                }

                "--test" => {
                    let value = args_iter.next().ok_or("Missing value for test command")?;
                    test_cmd = Some(value);
                }

                "--src-json" => {
                    let value = args_iter
                        .next()
                        .ok_or("Missing value for JSON data source")?;
                    data_source = Some(DataSource::JsonFile(value));
                }

                "--src-cmd" => {
                    let value = args_iter
                        .next()
                        .ok_or("Missing value for command data source")?;
                    data_source = Some(DataSource::Command(value));
                }

                "--help" => {
                    println!("Usage: boast-cli [OPTIONS]");
                    println!();
                    println!("Options:");
                    println!("  -q, --confidence <FLOAT>       Confidence level (0.0 to 1.0)");
                    println!("  -p, --outlier_rate <FLOAT>    Outlier rate (0.0 to 1.0)");
                    println!("      --pass_ratio <FLOAT>      Pass ratio (0.0 to 1.0)");
                    println!("      --timeout <INT>           Timeout in seconds");
                    println!("      --test <CMD>              Command to run the test");
                    println!("      --src-json <FILE>         JSON file as data source");
                    println!("      --src-cmd <CMD>          Command as data source");
                    println!("      --help                   Print this help message");
                    std::process::exit(0);
                }

                _ => return Err("Unknown argument"),
            }
        }

        let q = q.ok_or("Missing required argument: confidence (or q)")?;
        let p = p.ok_or("Missing required argument: outlier_rate (or p)")?;
        let data_source = data_source
            .ok_or("Missing required argument: data source (--src-json or --src-cmd)")?;
        let test_cmd = test_cmd.ok_or("Missing required argument: test command (--test)")?;

        Ok(CliArgs {
            q,
            p,
            pass_ratio,
            timeout,
            data_source,
            test_cmd,
        })
    }
}

fn main() {
    let args = match CliArgs::parse() {
        Ok(args) => args,
        Err(err) => {
            eprintln!("Error parsing arguments: {}", err);
            std::process::exit(1);
        }
    };

    let data_source = match args.data_source.load_data() {
        Ok(ds) => ds,
        Err(err) => {
            eprintln!("Error loading data source: {}", err);
            std::process::exit(1);
        }
    };

    let test_cmd_parts = args.test_cmd.split_whitespace().collect::<Vec<_>>();
    if test_cmd_parts.is_empty() {
        eprintln!("Test command cannot be empty");
        std::process::exit(1);
    }

    println!("{:?}", test_cmd_parts);

    let run_test = |data: &[String]| {
        let mut test_cmd = std::process::Command::new(test_cmd_parts[0]);
        if test_cmd_parts.len() > 1 {
            test_cmd.args(&test_cmd_parts[1..]);
        }

        // we will send the data as a single JSON string argument to stdin of the test command
        let data_json = serde_json::to_string(&data).expect("Failed to serialize data to JSON");

        let output = test_cmd.spawn().expect("Failed to spawn test command");

        let mut stdin = output.stdin.as_ref().expect("Failed to open stdin");
        stdin
            .write_all(data_json.as_bytes())
            .expect("Failed to write data to test command stdin");

        // Now the return status of the test command will determine whether the test passed or failed
        let output = output
            .wait_with_output()
            .expect("Failed to wait on test command");
        if !output.status.success() {
            // Panic with the output of the command for debugging
            let stderr = String::from_utf8_lossy(&output.stderr);
            panic!("{stderr}");
        }
    };

    let mut options = boast::Options::new(args.q, args.p);
    if let Some(pass_ratio) = args.pass_ratio {
        options = options.with_pass_ratio(pass_ratio);
    }
    if let Some(timeout) = args.timeout {
        options = options.with_timeout(std::time::Duration::from_secs(timeout));
    }

    let runner = boast::Runner::new(options, data_source, run_test);
    runner.run();
}
