"""
CLEANUP SCRIPT - Remove Old 2K Trial Files
Run this to keep only your new 8K+ trial dataset
"""

import pandas as pd
from pathlib import Path
import os

print("\n" + "="*70)
print("CLINICAL TRIAL DATA CLEANUP")
print("="*70 + "\n")

# Find all processed files
processed_dir = Path('data/processed')

if not processed_dir.exists():
    print("❌ ERROR: data/processed/ directory not found")
    print("   Make sure you're in the project root directory")
    exit(1)

files = list(processed_dir.glob('clinical_trials_features*.csv'))

if not files:
    print("❌ ERROR: No processed files found")
    exit(1)

print(f"Found {len(files)} processed files\n")

# Categorize by size
small_files = []
large_files = []

for f in files:
    try:
        # Just check file size to avoid loading huge files
        size_mb = f.stat().st_size / 1024 / 1024
        
        # Files < 2MB are old 2K datasets, files > 4MB are new 8K datasets
        if size_mb < 2.0:
            # Quick verify
            df = pd.read_csv(f, nrows=100)
            full_df = pd.read_csv(f, low_memory=False)
            count = len(full_df)
            small_files.append((f, count))
            print(f"❌ OLD (2K dataset): {f.name} - {count:,} trials ({size_mb:.1f} MB)")
        else:
            # Quick verify
            df = pd.read_csv(f, nrows=100)
            full_df = pd.read_csv(f, low_memory=False)
            count = len(full_df)
            large_files.append((f, count))
            print(f"✅ NEW (8K dataset): {f.name} - {count:,} trials ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"⚠️  Error reading {f.name}: {e}")

print(f"\n{'='*70}")
print(f"Summary: {len(small_files)} old files, {len(large_files)} new files")
print(f"{'='*70}\n")

if len(small_files) == 0:
    print("✅ No old files to delete - you're all set!")
    if len(large_files) > 1:
        print(f"\nNote: You have {len(large_files)} copies of the 8K dataset")
        print("      Keeping all of them (they're the same data)")
    exit(0)

# Confirm deletion
print(f"This will DELETE {len(small_files)} old 2K trial files:")
for f, count in small_files:
    print(f"  - {f.name}")

response = input(f"\nDelete these {len(small_files)} files? (yes/no): ").strip().lower()

if response not in ['yes', 'y']:
    print("\n❌ Cleanup cancelled")
    exit(0)

# Delete old files
print(f"\n{'='*70}")
print("DELETING OLD FILES")
print(f"{'='*70}\n")

deleted = 0
for f, count in small_files:
    try:
        print(f"Deleting {f.name}...")
        f.unlink()
        deleted += 1
    except Exception as e:
        print(f"❌ Error deleting {f.name}: {e}")

print(f"\n✓ Deleted {deleted} old files")

# If multiple large files, keep only the newest
if len(large_files) > 1:
    print(f"\n{'='*70}")
    print("MULTIPLE 8K FILES FOUND")
    print(f"{'='*70}\n")
    
    # Sort by filename (timestamp in name)
    large_files.sort(key=lambda x: x[0].name, reverse=True)
    keep_file = large_files[0][0]
    
    print(f"Keeping newest: {keep_file.name} ({large_files[0][1]:,} trials)")
    print(f"\nOther copies:")
    for f, count in large_files[1:]:
        print(f"  - {f.name}")
    
    response = input(f"\nDelete {len(large_files)-1} duplicate copies? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        for f, count in large_files[1:]:
            try:
                print(f"Deleting {f.name}...")
                f.unlink()
            except Exception as e:
                print(f"❌ Error deleting {f.name}: {e}")

# Final verification
print(f"\n{'='*70}")
print("FINAL STATE")
print(f"{'='*70}\n")

remaining = list(processed_dir.glob('clinical_trials_features*.csv'))
print(f"Files remaining: {len(remaining)}")

for f in remaining:
    try:
        df = pd.read_csv(f, low_memory=False)
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  ✅ {f.name}")
        print(f"     Trials: {len(df):,}")
        print(f"     Size: {size_mb:.1f} MB")
        print()
    except Exception as e:
        print(f"  ⚠️  {f.name}: Error - {e}")

print(f"{'='*70}")
print("✅ CLEANUP COMPLETE!")
print(f"{'='*70}\n")

if len(remaining) == 1:
    print("Your app will now load the 8K+ trial dataset!")
    print("\nNext steps:")
    print("  1. Restart Streamlit: streamlit run src/app/streamlit_app.py")
    print("  2. Verify Overview page shows 8,471 trials")
    print("  3. Deploy: git add data/ && git commit && git push")
else:
    print(f"Warning: {len(remaining)} files remaining")
    print("App will load the most recent one")

print(f"\n{'='*70}\n")
