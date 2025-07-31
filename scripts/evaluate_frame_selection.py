import glob

val_scan_names = set([scan.replace('.mha', '') for scan in val_scans])

def process_val_scans(scan_dir, model, transform, threshold=0.4):
    results = []
    all_scan_paths = sorted(glob.glob(os.path.join(scan_dir, "*.mha")))
    for path in tqdm(all_scan_paths, desc="Processing validation scans"):
        scan_id = os.path.basename(path).replace('.mha', '')
        if scan_id not in val_scan_names:
            continue
        try:
            frames_tensor = load_mha_frames(path, transform)
            probs = score_frames(model, frames_tensor)
            scores = probs[:, 1]  # class 1 = optimal
            max_score = np.max(scores)
            best_idx = int(np.argmax(scores)) if max_score >= threshold else -1
            results.append({"scan": scan_id + ".mha", "best_frame": best_idx, "score": max_score})
        except Exception as e:
            print(f"Error processing {scan_id}: {e}")
    return pd.DataFrame(results)

scan_folder = "/content/drive/MyDrive/acouslic-ai-train-set/images/stacked_fetal_ultrasound"
df_results = process_val_scans(scan_folder, model, val_test_transform)
df_results.to_csv("/content/drive/MyDrive/best_frames_val_only.csv", index=False)
