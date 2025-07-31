PREDICTIONS_CSV = "/content/drive/MyDrive/best_frames_val_only.csv"
MASK_DIR = "/content/drive/MyDrive/acouslic-ai-train-set/masks/stacked_fetal_abdomen"

pred_df = pd.read_csv(PREDICTIONS_CSV)
scores = []

def get_frame_labels_from_mask(mask_path):
    mask = sitk.ReadImage(mask_path)
    mask_array = sitk.GetArrayFromImage(mask)
    labels = []
    for frame in mask_array:
        unique = np.unique(frame)
        if 1 in unique: labels.append("optimal")
        elif 2 in unique: labels.append("suboptimal")
        else: labels.append("irrelevant")
    return labels

for _, row in tqdm(pred_df.iterrows(), total=len(pred_df)):
    scan_id = row["scan"]
    selected = int(row["best_frame"])
    mask_path = os.path.join(MASK_DIR, scan_id)
    if not os.path.exists(mask_path): continue
    frame_labels = get_frame_labels_from_mask(mask_path)
    if selected == -1 or selected >= len(frame_labels): scores.append(0.0); continue
    label = frame_labels[selected]
    has_optimal = "optimal" in frame_labels
    if label == "optimal": scores.append(1.0)
    elif label == "suboptimal" and has_optimal: scores.append(0.6)
    else: scores.append(0.0)

wfss_score = np.mean(scores)
print(f"WFSS: {wfss_score:.4f}")
