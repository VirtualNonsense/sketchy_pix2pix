use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::blocking::Client;
use std::fs::{create_dir_all, File};
use std::io::{Read, Write};  
use std::path::PathBuf;
use std::str::FromStr;
use std::time::Duration;
use sevenz_rust2::decompress_file;



#[derive(Parser)]
#[command(version, about = "Ein Hilfstool für Build- und Dev-Aufgaben")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// downloads the sketchy db from the google drive
    DownloadSketchyDB {
               
        #[arg(default_value = "https://drive.usercontent.google.com/download?id=0B7ISyeE8QtDdTjE1MG9Gcy1kSkE&export=download&authuser=0&resourcekey=0-r6nB4crmdU-LK7H38xnOUw&confirm=t&uuid=4da267aa-44c3-41b4-981b-0fc230e9b4d6&at=AEz70l4NZ8fecy0LI--8bbeLDGOY%3A1740916873747")]
        url: String,
        #[arg(default_value = "data/sketchydb_256x256/sketchydb_256x256.7z")]
        output: String,
    },
    /// Extracts the zip file downloaded before to a given path
    ExtractSketchyDB {
        #[arg(default_value = "data/sketchydb_256x256/sketchydb_256x256.7z")]
        zip_path: String,
        #[arg(default_value = "data/sketchydb_256x256")]
        dest: String,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::DownloadSketchyDB { url, output } => {
                        println!("🔽 downloading {} to {}", url, output);
                        match download_zip(&url, &output) {
                            Ok(_) => println!("✅ Download successful: {}", output),
                            Err(e) => eprintln!("❌ error: {}", e),
                        }
            }

        Commands::ExtractSketchyDB { zip_path, dest } => {
            println!("📦 Extracting: {} to {}", zip_path, dest);
            match extract_7z(&zip_path, &dest) {
                Ok(_) => println!("✅ Successfully excracted {} to: {}", zip_path, dest),
                Err(e) => eprintln!("❌ Error: {}", e),
            }
        }
    }
}

fn download_zip(url: &str, output: &str) -> Result<(), Box<dyn std::error::Error>> {
    let file = PathBuf::from_str(output).expect("failed to parse output directory");
    let parent_res = file.parent();
    if let Some(parent) = parent_res {
        create_dir_all(parent)?;
    }
    let client = Client::builder()
        .timeout(Duration::from_secs(300))
        .build()?;

    let mut response = client.get(url).send()?;
    
    if !response.status().is_success() {
        return Err(format!(": {}", response.status()).into());
    }

    let total_size = response.content_length().unwrap_or(0);
    let pb = ProgressBar::new(total_size);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{wide_bar} {bytes}/{total_bytes} ({eta})")?
            .progress_chars("█▉▊▋▌▍▎▏  "),
    );

    let mut dest = File::create(output)?;
    let mut buffer = [0; 8192]; // 8 KB Puffer
    let mut downloaded = 0;

    while let Ok(n) = response.read(&mut buffer) {
        if n == 0 {
            break;
        }
        dest.write_all(&buffer[..n])?;
        downloaded += n as u64;
        pb.set_position(downloaded);
    }

    pb.finish_with_message("✅ Download finished!");
    Ok(())
}



/// Extracts a 7z file using `sevenz-rs2`
fn extract_7z(archive_path: &str, dest: &str) -> Result<(), Box<dyn std::error::Error>> {
    create_dir_all(dest)?;

    println!("📦 Extracting 7z archive to: {}", dest);
    println!("⌛ This may take a while...");
    
    decompress_file(archive_path, dest)?;

    println!("✅ 7z extraction completed!");
    Ok(())
}

