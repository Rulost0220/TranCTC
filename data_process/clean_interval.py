import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
if __name__ == "__main__":
	# dir = 'test_flow/'
	# dir_path = Path(dir)
	# if not dir_path.is_dir():
	# English note retained from the original workflow.
	# count_total = 0
	# for file in tqdm(sorted(dir_path.rglob("*.csv"))):
	# 	try:
	# 		df = pd.read_csv(file)
	# 		count = len(df) - 19 + 1
	# 		count_total += count
	# 		print(f"{file}: {count}")
	# 	except Exception as e:
	# English note retained from the original workflow.
	# 		continue
	
	# English note retained from the original workflow.

	# dir = '../npy_data/npy_19_11/test/'
	# dir_path = Path(dir)
	# if not dir_path.is_dir():
	# English note retained from the original workflow.
	# count_total = 0
	# English note retained from the original workflow.


	dir = './test_dir/'
	dir_path = Path(dir)
	if not dir_path.is_dir():
		raise ValueError(f"{dir_path} is not a valid directory")

	for file in tqdm(sorted(dir_path.rglob("*.csv"))):
		try:
			# English note retained from the original workflow.
			df = pd.read_csv(file)
			
			# English note retained from the original workflow.
			original_rows = len(df)
			
			# English note retained from the original workflow.
			df = df[(df['interval'] != 0.0) & (df['interval'] < 1000.0)]
			
			# English note retained from the original workflow.
			new_rows = len(df)
			
			# English note retained from the original workflow.
			if new_rows < original_rows:
				deleted_count = original_rows - new_rows
				print(f"Processed: {file}, removed {deleted_count} anomalous rows")
				
				# English note retained from the original workflow.
				df.to_csv(file, index=False)
				
			# English note retained from the original workflow.
			else:
				print(f"Checked: {file}, no anomalous rows")
				
		except Exception as e:
			print(f"[skip] processing failed {file}: {e}")
			continue
		
