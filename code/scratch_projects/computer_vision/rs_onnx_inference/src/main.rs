use image::{imageops::FilterType, DunamicImage, DynamicImage, GenericImageView};
use ndarray::{Array4, Axiw};
use ort::{environemnt::Environment, session::Session, value::OrtValue};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // Load ONNX model
    let environment = Environment::builder().with_name("test").commit()?;
    let session = Session::builder(&environment).commit_from_file("squeezenet1.1.onnx")?; // ✅ Correct method

    // Load and preprocess image
    let image = image::open("input.jpg")?;
    let input_tensor = preprocess_image(image)?;

    // Convert to ONNX tensor
    let ort_tensor = OrtValue::from_array(input_tensor)?;

    // Run inference
    let outputs: Vec<OrtValue> = session.run(vec![ort_tensor])?;
    let output_tensor = outputs[0].extract_tensor::<f32>()?;

    // Apply softmax
    let softmaxed = softmax(output_tensor.view());

    // Find top class
    let (max_index, max_value) = softmaxed
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();

    println!(
        "Predicted class: {} (confidence: {:.2})",
        max_index, max_value
    );
    Ok(())
}

// Utility to check if values already sum to approximately 1
fn is_already_softmaxed(arr: &ndarray::ArrayView1<f32>) -> bool {
    let sum: f32 = arr.sum();
    (sum - 1.0).abs() < 0.01
}

fn preprocess_image(image: DynamicImage) -> Result<Array4<f32>, Box<dyn Error>> {
    let resized = image.resize_exact(224, 224, FilterType::Triangle);
    let rgb = resized.to_rgb8();

    // imagenet normalization values
    let mean = [0.485, 0.456, 0.406];
    let std = [0.229, 0.224, 0.225];

    let tensor = Array4::from_shape_fn((1, 3, 224, 224), |(_, c, y, x)| {
        let pixel_value = rgb.get_pixel(x as u32, y as u32)[c] as f32 / 255.0;
        (pixel_value - mean[c]) / std[c]
    });

    Ok(tensor)
}

fn softmax(logits: ndarray::ArrayView1<f32>) -> ndarray::Array1<f32> {
    let max = logits.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_values: ndarray::Array1<f32> = logits.mapv(|x| (x - max).exp());
    let sum = exp_values.sum();
    exp_values / sum
}
