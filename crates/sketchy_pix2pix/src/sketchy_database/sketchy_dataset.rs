use burn::{data::dataset::Dataset, tensor::TensorData};
use image;
use log::info;
use rand::Rng;
use ron::de::{SpannedError, from_reader};
use ron::ser::to_writer;
use serde::{Deserialize, Serialize};
use std::fs;
use std::{
    fs::File,
    io,
    path::{Path, PathBuf},
};
use thiserror::Error;
use walkdir::WalkDir;

pub enum PhotoAugmentation {
    /// image is non-uniformly scaled to 256x256
    Normal,
    /// image bounding box scaled to 256x256 with
    /// an additional +10% on each edge; note
    /// that due to position within the image,
    ///  sometimes the object is not centered
    ScaledAndCentered,
}

impl PhotoAugmentation {
    pub fn get_folder_name(&self) -> &str {
        match self {
            PhotoAugmentation::Normal => "tx_000000000000",
            PhotoAugmentation::ScaledAndCentered => "tx_000100000000",
        }
    }
}

pub enum SketchAugmentation {
    /// sketch canvas is rendered to 256x256
    /// such that it undergoes the same
    /// scaling as the paired photo
    Normal,
    /// sketch is centered and uniformly scaled
    /// such that its greatest dimension (x or y)
    /// fills 78% of the canvas (roughly the same
    /// as in Eitz 2012 sketch data set)
    ScaledAndCentered,
    /// sketch is translated such that it is
    /// centered on the object bounding box
    Centered,
    /// sketch is centered on bounding box and
    /// is uniformly scaled such that one dimension
    /// (x or y; whichever requires the least amount
    /// of scaling) fits within the bounding box
    CenteredAndMinScalled,
    /// sketch is centered on bounding box and
    /// is uniformly scaled such that one dimension
    /// (x or y; whichever requires the most amount
    /// of scaling) fits within the bounding box
    CenteredAndMaxScalled,
    /// sketch is centered on bounding box and
    /// is non-uniformly scaled such that it
    /// completely fits within the bounding box
    CenteredNonuniformScaling,
}
impl SketchAugmentation {
    pub fn get_folder_name(&self) -> &str {
        match self {
            SketchAugmentation::Normal => "tx_000000000000",
            SketchAugmentation::ScaledAndCentered => "tx_000100000000",
            SketchAugmentation::Centered => "tx_000000000010",
            SketchAugmentation::CenteredAndMinScalled => "tx_000000000110",
            SketchAugmentation::CenteredAndMaxScalled => "tx_000000001010",
            SketchAugmentation::CenteredNonuniformScaling => "tx_000000001110",
        }
    }
}
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
pub enum SketchyClass {
    Airplane,
    AlarmClock,
    Ant,
    Ape,
    Apple,
    Armor,
    Axe,
    Banana,
    Bat,
    Bear,
    Bee,
    Beetle,
    Bell,
    Bench,
    Bicycle,
    Blimp,
    Bread,
    Butterfly,
    Cabin,
    Camel,
    Candle,
    Cannon,
    CarSedan,
    Castle,
    Cat,
    Chair,
    Chicken,
    Church,
    Couch,
    Cow,
    Crab,
    Crocodilian,
    Cup,
    Deer,
    Dog,
    Dolphin,
    Door,
    Duck,
    Elephant,
    Eyeglasses,
    Fan,
    Fish,
    Flower,
    Frog,
    Geyser,
    Giraffe,
    Guitar,
    Hamburger,
    Hammer,
    Harp,
    Hat,
    Hedgehog,
    Helicopter,
    HermitCrab,
    Horse,
    HotAirBalloon,
    Hotdog,
    Hourglass,
    JackOLantern,
    Jellyfish,
    Kangaroo,
    Knife,
    Lion,
    Lizard,
    Lobster,
    Motorcycle,
    Mouse,
    Mushroom,
    Owl,
    Parrot,
    Pear,
    Penguin,
    Piano,
    PickupTruck,
    Pig,
    Pineapple,
    Pistol,
    Pizza,
    Pretzel,
    Rabbit,
    Raccoon,
    Racket,
    Ray,
    Rhinoceros,
    Rifle,
    Rocket,
    Sailboat,
    Saw,
    Saxophone,
    Scissors,
    Scorpion,
    Seagull,
    Seal,
    SeaTurtle,
    Shark,
    Sheep,
    Shoe,
    Skyscraper,
    Snail,
    Snake,
    Songbird,
    Spider,
    Spoon,
    Squirrel,
    Starfish,
    Strawberry,
    Swan,
    Sword,
    Table,
    Tank,
    Teapot,
    TeddyBear,
    Tiger,
    Tree,
    Trumpet,
    Turtle,
    Umbrella,
    Violin,
    Volcano,
    WadingBird,
    Wheelchair,
    Windmill,
    Window,
    WineBottle,
    Zebra,
}

impl TryFrom<&str> for SketchyClass {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "airplane" => Ok(Self::Airplane),
            "alarm_clock" => Ok(Self::AlarmClock),
            "ant" => Ok(Self::Ant),
            "ape" => Ok(Self::Ape),
            "apple" => Ok(Self::Apple),
            "armor" => Ok(Self::Armor),
            "axe" => Ok(Self::Axe),
            "banana" => Ok(Self::Banana),
            "bat" => Ok(Self::Bat),
            "bear" => Ok(Self::Bear),
            "bee" => Ok(Self::Bee),
            "beetle" => Ok(Self::Beetle),
            "bell" => Ok(Self::Bell),
            "bench" => Ok(Self::Bench),
            "bicycle" => Ok(Self::Bicycle),
            "blimp" => Ok(Self::Blimp),
            "bread" => Ok(Self::Bread),
            "butterfly" => Ok(Self::Butterfly),
            "cabin" => Ok(Self::Cabin),
            "camel" => Ok(Self::Camel),
            "candle" => Ok(Self::Candle),
            "cannon" => Ok(Self::Cannon),
            "car_(sedan)" => Ok(Self::CarSedan),
            "castle" => Ok(Self::Castle),
            "cat" => Ok(Self::Cat),
            "chair" => Ok(Self::Chair),
            "chicken" => Ok(Self::Chicken),
            "church" => Ok(Self::Church),
            "couch" => Ok(Self::Couch),
            "cow" => Ok(Self::Cow),
            "crab" => Ok(Self::Crab),
            "crocodilian" => Ok(Self::Crocodilian),
            "cup" => Ok(Self::Cup),
            "deer" => Ok(Self::Deer),
            "dog" => Ok(Self::Dog),
            "dolphin" => Ok(Self::Dolphin),
            "door" => Ok(Self::Door),
            "duck" => Ok(Self::Duck),
            "elephant" => Ok(Self::Elephant),
            "eyeglasses" => Ok(Self::Eyeglasses),
            "fan" => Ok(Self::Fan),
            "fish" => Ok(Self::Fish),
            "flower" => Ok(Self::Flower),
            "frog" => Ok(Self::Frog),
            "geyser" => Ok(Self::Geyser),
            "giraffe" => Ok(Self::Giraffe),
            "guitar" => Ok(Self::Guitar),
            "hamburger" => Ok(Self::Hamburger),
            "hammer" => Ok(Self::Hammer),
            "harp" => Ok(Self::Harp),
            "hat" => Ok(Self::Hat),
            "hedgehog" => Ok(Self::Hedgehog),
            "helicopter" => Ok(Self::Helicopter),
            "hermit_crab" => Ok(Self::HermitCrab),
            "horse" => Ok(Self::Horse),
            "hot-air_balloon" => Ok(Self::HotAirBalloon),
            "hotdog" => Ok(Self::Hotdog),
            "hourglass" => Ok(Self::Hourglass),
            "jack-o-lantern" => Ok(Self::JackOLantern),
            "jellyfish" => Ok(Self::Jellyfish),
            "kangaroo" => Ok(Self::Kangaroo),
            "knife" => Ok(Self::Knife),
            "lion" => Ok(Self::Lion),
            "lizard" => Ok(Self::Lizard),
            "lobster" => Ok(Self::Lobster),
            "motorcycle" => Ok(Self::Motorcycle),
            "mouse" => Ok(Self::Mouse),
            "mushroom" => Ok(Self::Mushroom),
            "owl" => Ok(Self::Owl),
            "parrot" => Ok(Self::Parrot),
            "pear" => Ok(Self::Pear),
            "penguin" => Ok(Self::Penguin),
            "piano" => Ok(Self::Piano),
            "pickup_truck" => Ok(Self::PickupTruck),
            "pig" => Ok(Self::Pig),
            "pineapple" => Ok(Self::Pineapple),
            "pistol" => Ok(Self::Pistol),
            "pizza" => Ok(Self::Pizza),
            "pretzel" => Ok(Self::Pretzel),
            "rabbit" => Ok(Self::Rabbit),
            "raccoon" => Ok(Self::Raccoon),
            "racket" => Ok(Self::Racket),
            "ray" => Ok(Self::Ray),
            "rhinoceros" => Ok(Self::Rhinoceros),
            "rifle" => Ok(Self::Rifle),
            "rocket" => Ok(Self::Rocket),
            "sailboat" => Ok(Self::Sailboat),
            "saw" => Ok(Self::Saw),
            "saxophone" => Ok(Self::Saxophone),
            "scissors" => Ok(Self::Scissors),
            "scorpion" => Ok(Self::Scorpion),
            "seagull" => Ok(Self::Seagull),
            "seal" => Ok(Self::Seal),
            "sea_turtle" => Ok(Self::SeaTurtle),
            "shark" => Ok(Self::Shark),
            "sheep" => Ok(Self::Sheep),
            "shoe" => Ok(Self::Shoe),
            "skyscraper" => Ok(Self::Skyscraper),
            "snail" => Ok(Self::Snail),
            "snake" => Ok(Self::Snake),
            "songbird" => Ok(Self::Songbird),
            "spider" => Ok(Self::Spider),
            "spoon" => Ok(Self::Spoon),
            "squirrel" => Ok(Self::Squirrel),
            "starfish" => Ok(Self::Starfish),
            "strawberry" => Ok(Self::Strawberry),
            "swan" => Ok(Self::Swan),
            "sword" => Ok(Self::Sword),
            "table" => Ok(Self::Table),
            "tank" => Ok(Self::Tank),
            "teapot" => Ok(Self::Teapot),
            "teddy_bear" => Ok(Self::TeddyBear),
            "tiger" => Ok(Self::Tiger),
            "tree" => Ok(Self::Tree),
            "trumpet" => Ok(Self::Trumpet),
            "turtle" => Ok(Self::Turtle),
            "umbrella" => Ok(Self::Umbrella),
            "violin" => Ok(Self::Violin),
            "volcano" => Ok(Self::Volcano),
            "wading_bird" => Ok(Self::WadingBird),
            "wheelchair" => Ok(Self::Wheelchair),
            "windmill" => Ok(Self::Windmill),
            "window" => Ok(Self::Window),
            "wine_bottle" => Ok(Self::WineBottle),
            "zebra" => Ok(Self::Zebra),
            anything_else => Err(format!("{} is not a supported class", anything_else)),
        }
    }
}

// Repräsentiert ein Foto-Skizzen-Paar (Dateipfade oder geladene Pixelwerte)
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct SketchyItem {
    pub sketch_class: SketchyClass,
    pub photo: PathBuf,
    pub sketch: PathBuf,
}

#[derive(Error, Debug)]
pub enum ImageError {
    #[error("failed to load image due to {:?}", .0)]
    LoadingError(#[from] io::Error),
    #[error("Failed to decode image due to {:?}", .0)]
    DecodingError(#[from] image::error::ImageError),
}

impl SketchyItem {
    fn open_picture(path: &Path, bw: bool) -> Result<TensorData, ImageError> {
        let image_result = image::ImageReader::open(path);
        if let Err(e) = image_result {
            return Err(e.into());
        }
        let decoding_result = image_result.unwrap().decode();
        if let Err(e) = decoding_result {
            return Err(e.into());
        }

        let mut image = decoding_result.unwrap();
        let height = image.height() as usize;
        let width = image.width() as usize;
        let channel: usize = if bw {
            1
        } else {
            3
        };
        let batch: usize = 1;
        if bw{
            image = image.grayscale()
        }
        let bytes = image.into_bytes();
        Ok(TensorData::new(bytes, vec![batch, height, width, channel]))
    }

    pub fn load_photo(&self) -> Result<TensorData, ImageError> {
        return Self::open_picture(&self.photo, false);
    }

    pub fn load_sketch(&self) -> Result<TensorData, ImageError> {
        return Self::open_picture(&self.sketch, true);
    }
}

#[derive(Error, Debug)]
pub enum SketchyDatasetError {
    #[error("The path {} is not valid because {}", .path, .reason)]
    InvalidPath { path: String, reason: String },

    #[error("Unable to parse the name of {}", .0)]
    NameParsingError(String),
    #[error("Unable to deserialize ron file due to {:?}", .0)]
    RonDeserializationError(#[from] SpannedError),
    #[error("Unable load ron file due to {:?}", .0)]
    RonFileLoadingError(#[from] io::Error),
    #[error("Unable to serialize ron file due to {:?}", .0)]
    RonSerializationError(#[from] ron::error::Error),
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub struct SketchyDataset {
    name: String,
    items: Vec<SketchyItem>,
}

impl SketchyDataset {
    pub fn new(
        photo_dir: &Path,
        sketch_dir: &Path,
        photo_mode: PhotoAugmentation,
        sketch_mode: SketchAugmentation,
    ) -> Result<Self, SketchyDatasetError> {
        fn check_dir(path: &Path) -> bool {
            path.exists() && path.is_dir()
        }

        if !check_dir(photo_dir) {
            return Err(SketchyDatasetError::InvalidPath {
                path: format!("{:?}", photo_dir),
                reason: "the photo path path does not exist or is not a directory".into(),
            });
        }
        if !check_dir(sketch_dir) {
            return Err(SketchyDatasetError::InvalidPath {
                path: format!("{:?}", sketch_dir),
                reason: "the sketch path does not exist or is not a directory".into(),
            });
        }
        let photo_dir = photo_dir.join(photo_mode.get_folder_name());
        let sketch_dir = sketch_dir.join(sketch_mode.get_folder_name());

        if !check_dir(&photo_dir) {
            return Err(SketchyDatasetError::InvalidPath {
                path: format!("{:?}", photo_dir),
                reason: "the photo path does not exist or is not a directory".into(),
            });
        }
        if !check_dir(&sketch_dir) {
            return Err(SketchyDatasetError::InvalidPath {
                path: format!("{:?}", sketch_dir),
                reason: "the sketch path does not exist or is not a directory".into(),
            });
        }
        let mut entries = vec![];
        for class_directory in WalkDir::new(&photo_dir)
            .into_iter()
            .flat_map(|t| t.ok())
            .filter(|e| e.file_type().is_dir())
        {
            let class_name = class_directory.file_name();

            let class_str = class_name.to_str();
            if class_str.is_none() {
                info!(
                    "{:?} does not a string representation and will be skipped",
                    class_name
                );
                continue;
            }
            let class_str = class_str.unwrap();
            let class_enum: Result<SketchyClass, String> = class_str.try_into();
            if let Err(e) = class_enum {
                info!("{e}");
                continue;
            }
            let class_enum = class_enum.unwrap();

            for photo in WalkDir::new(&class_directory.path())
                .into_iter()
                .flat_map(Result::ok)
                .filter(|e| !e.file_type().is_dir())
            {
                let photo_path = photo.path().to_owned();
                let mut sketch_paths = vec![];
                if let Some(photo_name) = photo_path.file_stem() {
                    let photo_name = photo_name.to_str().unwrap();
                    for sketch in WalkDir::new(sketch_dir.join(class_name))
                        .into_iter()
                        .flat_map(Result::ok)
                        .filter(|e| {
                            if let Some(name) = e.path().file_stem() {
                                let name = name.to_str().unwrap();
                                name.starts_with(photo_name)
                            } else {
                                false
                            }
                        })
                    {
                        sketch_paths.push(sketch.path().to_owned());
                    }
                } else {
                    return Err(SketchyDatasetError::NameParsingError(format!(
                        "{:?}",
                        photo_path
                    )));
                }
                if sketch_paths.len() == 0 {
                    continue;
                }

                for skech in sketch_paths{
                    entries.push(SketchyItem {
                        sketch_class: class_enum,
                        photo: photo_path.clone(),
                        sketch: skech,
                    });
                }
            }
        }

        Ok(Self {
            name: "unsplit".into(),
            items: entries,
        })
    }

    pub fn save_to_ron(&self, path: &Path) -> Result<(), SketchyDatasetError> {
        if path.exists(){
            if path.is_dir(){
                return Err(SketchyDatasetError::InvalidPath {
                    path: format!("{:?}", path),
                    reason: "Invalid Path. Please enter a .ron path".into(),
                });
            }
            if let Err(e) = fs::remove_file(path){
                info!("Removing {:?}", path);
                return Err(e.into())
            };
        }
        let file_result = File::create_new(path);
        if let Err(er) = file_result {
            return Err(er.into());
        }

        let t = to_writer(file_result.unwrap(), self);
        if let Err(e) = t {
            return Err(e.into());
        }
        Ok(())
    }
    pub fn load_from_ron(path: &Path) -> Result<Self, SketchyDatasetError> {
        if !path.exists() {
            return Err(SketchyDatasetError::InvalidPath {
                path: format!("{:?}", path),
                reason: "The path does not exist".into(),
            });
        }
        if path.is_dir() {
            return Err(SketchyDatasetError::InvalidPath {
                path: format!("{:?}", path),
                reason: "The path does not lead to a valid .ron file".into(),
            });
        }

        let file_result = File::open(path);
        if let Err(er) = file_result {
            return Err(er.into());
        }

        let r = match from_reader(file_result.unwrap()) {
            Ok(ds) => Ok(ds),
            Err(er) => Err(er.into()),
        };
        r
    }
    pub fn split(mut self, train_split_ratio: f64) -> (Self, Self){
        assert!(0. < train_split_ratio && train_split_ratio < 1., "The ratio must in (0, 1)");

        let train_amount = (train_split_ratio * self.len() as f64) as usize;
        let mut rng = rand::rng();
        let mut train_items = vec![];
        while train_items.len() < train_amount && self.items.len() > 0 {
            let idx = rng.random_range(0..self.items.len());
            train_items.push(self.items.swap_remove(idx));
        }
        (
            Self{
                name:"train".into(),
                items: train_items
            },
            Self{
                name:"test".into(),
                items: self.items
            },
        )
    }
    pub fn filter<P>(self, mut predicate: P) -> Self 
    where 
        P: FnMut(&SketchyItem) -> bool
    {
        Self{
            name: self.name.clone(),
            items: self.items.iter().filter(|item| predicate(item)).cloned().collect()
        }
    }

}

// Implementierung des Burn Dataset-Traits
impl Dataset<SketchyItem> for SketchyDataset {
    fn len(&self) -> usize {
        self.items.len()
    }

    fn get(&self, index: usize) -> Option<SketchyItem> {
        self.items.get(index).cloned()
    }
}
