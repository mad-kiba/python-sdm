def load_occurrences(df, lon_col, lat_col):
    """Загружает CSV с наблюдениями, фильтрует некорректные координаты."""
    #df = pd.read_csv(csv_path, sep=";")
    df[lat_col] = df[lat_col].astype(float)
    df[lon_col] = df[lon_col].astype(float)
    if lon_col not in df.columns or lat_col not in df.columns:
        raise ValueError(f"В CSV нет столбцов {lon_col}/{lat_col}")
    df = df[[lon_col, lat_col]].dropna()
    # Базовая фильтрация координат
    df = df[(df[lon_col] >= -180) & (df[lon_col] <= 180) & (df[lat_col] >= -90) & (df[lat_col] <= 90)]
    df = df.reset_index(drop=True)
    return df