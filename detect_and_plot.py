import boto3
import matplotlib.pyplot as plt

BUCKET = "rekognition-demo-anish"
IMAGE = "images/cats-vs.-dogs-group-scaled.jpeg"

rekognition = boto3.client(
    "rekognition",
    region_name="us-east-2"  # <-- use your bucket region
)


response = rekognition.detect_labels(
    Image={"S3Object": {"Bucket": BUCKET, "Name": IMAGE}},
    MaxLabels=10,
    MinConfidence=70
)

labels = []
confidences = []

for label in response["Labels"]:
    labels.append(label["Name"])
    confidences.append(label["Confidence"])

# Print output
print("\nDetected Labels:")
for l, c in zip(labels, confidences):
    print(f"{l}: {c:.2f}%")

# Plot
plt.figure(figsize=(10, 5))
plt.bar(labels, confidences)
plt.xlabel("Labels")
plt.ylabel("Confidence (%)")
plt.title("Amazon Rekognition – Image Label Confidence")
plt.xticks(rotation=45, ha="right")
plt.ylim(0, 100)
plt.tight_layout()
plt.show()
