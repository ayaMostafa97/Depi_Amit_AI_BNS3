import pandas as pd

# 1️⃣ Load data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print("✅ Data Loaded Successfully!")
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

# 2️⃣ Check data types
def check_dtypes(df):
    print("\n📝 Data Types of Columns:")
    print(df.dtypes)

# 3️⃣ Convert data types (optional specific column or auto all)
def convert_dtypes(df, column=None, dtype=None):
    if column and dtype:
        df[column] = df[column].astype(dtype)
        print(f"✅ Converted column '{column}' to {dtype}")
    else:
        df = df.convert_dtypes()
        print("✅ All columns converted to best possible dtypes")
    return df

# 4️⃣ Check missing values
def check_missing_values(df):
    print("\n🔍 Missing Values per Column:")
    print(df.isnull().sum())

# 5️⃣ Handle missing values
def handle_missing_values(df, method="mean"):
    """
    method: 'mean', 'median', 'mode', or 'drop'
    """
    for column in df.columns:
        if df[column].isnull().sum() > 0:
            # ✅ لو العمود رقمي
            if pd.api.types.is_numeric_dtype(df[column]):
                if method == "mean":
                    df[column] = df[column].fillna(df[column].mean())
                elif method == "median":
                    df[column] = df[column].fillna(df[column].median())
                elif method == "mode":
                    df[column] = df[column].fillna(df[column].mode()[0])
            else:
                # ✅ لو العمود نصي → نعوض بالـ mode
                df[column] = df[column].fillna(df[column].mode()[0])
    print(f"✅ Missing values handled using '{method}' method")
    return df