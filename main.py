import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import load_dataset
from itertools import islice
from IPython.display import display
from pathlib import Path

CATEGORY = "Electronics"
SAMPLE_SIZE = 100
DATASET_ID = "McAuley-Lab/Amazon-Reviews-2023"

REVIEWS_URL = (
    f"https://huggingface.co/datasets/{DATASET_ID}/resolve/main/"
    f"raw/review_categories/{CATEGORY}.jsonl"
)

def sample_jsonl(url, limit=SAMPLE_SIZE):
    """Stream JSON rows and keep only `limit` rows in memory."""
    dataset = load_dataset(
        "json", data_files={"train": url}, split="train", streaming=True
    )
    return pd.DataFrame(list(islice(dataset, limit)))

reviews_df = sample_jsonl(REVIEWS_URL)
print(f"Reviews loaded: {len(reviews_df)}")

display(reviews_df.head())

Items_To_Print = 10
pd.set_option("display.max_rows", Items_To_Print)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 100)
display(reviews_df.head(Items_To_Print))

METADATA_URL = (
    f"https://huggingface.co/datasets/{DATASET_ID}/resolve/main/"
    f"raw/meta_categories/meta_{CATEGORY}.jsonl"
)

item_metadata_df = sample_jsonl(METADATA_URL, SAMPLE_SIZE)
print(f"Metadata records loaded: {len(item_metadata_df)}")

METADATA_URL = (
    f"https://huggingface.co/datasets/{DATASET_ID}/resolve/main/"
    f"raw/meta_categories/meta_{CATEGORY}.jsonl"
)

item_metadata_df = sample_jsonl(METADATA_URL, SAMPLE_SIZE)
print(f"Metadata records loaded: {len(item_metadata_df)}")

display(item_metadata_df.head(10))

def safe_product_details(df, index):
    """Return a product title or none"""
    try:
        row = df.iloc[index]
        title = row["title"]
        price = pd.to_numeric(row["price"], errors="coerce")
        if not isinstance(title, str) or not title.strip():
            title = None
        if pd.isnull(price):
            price = None
        return {"title": title, "price": price}
    except (IndexError, KeyError, TypeError, ValueError):
        return None

print("Valid row:", safe_product_details(item_metadata_df, 0))
print("Invalid row:", safe_product_details(item_metadata_df, 999999))

print("Reviews columns:")
for column in reviews_df.columns:
    print("-", column)

print("\nMetadata columns:")
for column in item_metadata_df.columns:
    print("-", column)

print("Reviews data types:")
print(reviews_df.dtypes)
print("\nMetadata data types:")
print(item_metadata_df.dtypes)

print("Reviews summary:")
reviews_df.info()
print("\nMetadata summary:")
item_metadata_df.info()

# Work on a copy so the original sampled metadata stays unchanged.
cleaned_metadata_df = item_metadata_df.copy()

cleaned_metadata_df["price"] = pd.to_numeric(
    cleaned_metadata_df["price"], errors="coerce"
)
cleaned_metadata_df["average_rating"] = pd.to_numeric(
    cleaned_metadata_df["average_rating"], errors="coerce"
)

valid_title = cleaned_metadata_df["title"].apply(
    lambda title: isinstance(title, str) and bool(title.strip())
)
valid_rating = cleaned_metadata_df["average_rating"] >= 4.5
valid_price = cleaned_metadata_df["price"].notna()

top_products_df = cleaned_metadata_df.loc[
    valid_title & valid_rating & valid_price
].copy()

display(top_products_df.head())

products_count = len(item_metadata_df)
updated_products_count = len(top_products_df)
print("Products rows:", products_count)
print("Updated product rows:", updated_products_count)

def calculate_percentage(part, whole):
    """Return part as a percentage of whole, avoiding division by zero."""
    if whole == 0:
        return 0.0
    return (part / whole) * 100

retained_percentage = calculate_percentage(products_count, updated_products_count)
print(f"Retained: {retained_percentage:.2f}%")

csv_path = Path("top_products.csv")
parquet_path = Path("top_products.parquet")

top_products_df.to_csv(csv_path, index=False)
top_products_df.to_parquet(parquet_path, index=False)

for path in (csv_path, parquet_path):
    print(f"{path.resolve()} — {path.stat().st_size:,} bytes")

top_rated_titles = []
for title in top_products_df["title"]:
    top_rated_titles.append(title)

    print(top_rated_titles)

for number, title in enumerate(top_rated_titles, start=1):
    print(f"{number}. {title}")

if top_products_df.empty:
    print("No products matched the filters in this 100-row sample.")
else:
    plt.figure(figsize=(8, 5))
    ax = sns.histplot(data=top_products_df, x="price", kde=True, alpha=1)
    bar_colors = ("yellow", "blue", "red")
    visible_bars = (bar for bar in ax.patches if bar.get_height() > 0)
    for index, bar in enumerate(visible_bars):
        bar.set_facecolor(bar_colors[index % len(bar_colors)])
    plt.title("Prices of highly rated products")
    plt.xlabel("Price")
    plt.ylabel("Number of products")
    plt.tight_layout()
    plt.show()

if top_products_df.empty:
    print("No products available for the scatterplot.")
else:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=top_products_df, x="price", y="average_rating")
    plt.title("Product price vs. average rating")
    plt.xlabel("Price")
    plt.ylabel("Average rating")
    plt.tight_layout()
    plt.show()

if top_products_df.empty:
    print("No products available to plot.")
else:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.histplot(data=top_products_df, x="price", kde=True, ax=axes[0])
    axes[0].set_title("Price distribution")
    sns.scatterplot(
        data=top_products_df, x="price", y="average_rating", ax=axes[1]
    )
    axes[1].set_title("Price vs. rating")
    plt.tight_layout()
    plt.show()

if top_products_df.empty:
    print("No products available to plot.")
else:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.histplot(data=top_products_df, x="price", kde=True, ax=axes[0])
    axes[0].set_title("Price distribution")
    sns.scatterplot(
        data=top_products_df, x="price", y="average_rating", ax=axes[1]
    )
    axes[1].set_title("Price vs. rating")
    plt.tight_layout()
    plt.show()

average_rating_cutoff = 4.7
highly_rated_products_df = top_products_df.loc[
    top_products_df["average_rating"] >= average_rating_cutoff
].copy()

highly_rated_count = len(highly_rated_products_df)
highly_rated_percentage = calculate_percentage(
    highly_rated_count, len(top_products_df)
)
print(f"Products rated at least {average_rating_cutoff}: {highly_rated_count}")
print(f"Share of cleaned products: {highly_rated_percentage:.2f}%")

if highly_rated_products_df.empty:
    print("No products met the higher threshold in this sample.")
else:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=highly_rated_products_df, x="price", y="average_rating"
    )
    plt.title("Price vs. rating (4.7+ products)")
    plt.tight_layout()
    plt.show()

highly_rated_products_df_cutoff = 4.7

dashboard_df = top_products_df.loc[
    top_products_df["average_rating"] >= highly_rated_products_df_cutoff
].copy()

# Highest average rating first; cheaper product first for rating ties.
dashboard_df = dashboard_df.sort_values(
    by=["average_rating", "price"], ascending=[False, True]
)
print("Dashboard — top-rated products")
display(dashboard_df)

if dashboard_df.empty:
    print("No products to plot at this cutoff.")
else:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=dashboard_df, x="price", y="average_rating")
    plt.title("Dashboard: products rated 4.7+")
    plt.tight_layout()
    plt.show()


highly_rated_products_df_cutoff = 4.7

dashboard_df = top_products_df.loc[
    top_products_df["average_rating"] >= highly_rated_products_df_cutoff
].copy()

dashboard_df = dashboard_df.sort_values(
    by=["average_rating", "price"], ascending=[False, True]
)
print("Dashboard — top-rated products")
display(dashboard_df)

if dashboard_df.empty:
    print("No products to plot at this cutoff.")
else:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=dashboard_df, x="price", y="average_rating")
    plt.title("Dashboard: products rated 4.7+")
    plt.tight_layout()
    plt.show()

# Edit the reflection prompts below after inspecting the actual results.
conclusions = f"""Transformational AI — Data Engineering Exercise

Original metadata sample size: {products_count}
Cleaned sample size (rating >= 4.5, valid title and price): {updated_products_count}
Retained share: {retained_percentage:.2f}%
Products rated >= 4.7: {highly_rated_count}
Share of cleaned products rated >= 4.7: {highly_rated_percentage:.2f}%

Why cleaning mattered:
Numeric conversion changed invalid price/rating values into missing values,
so the filtering logic could handle them safely.

What the plots showed:
The biggest price bracket of highly rated products is 0 to 100, followed by
the 100 to 200 with a much smaller number (17 to 3). Interestingly, there 
are products with prices over 800 in the highest rated products but the 
price brackets between 200 and 800 are missing. The products with perfect
5.0 ratings are 2 between 0 and 100, 1 between 100 and 200, and 1 above 800.
The 4.5 rated products are all between 0 and 100.

Most useful Python technique:
The most useful Python technique is a for loop. In total, we have looped
10 times the products with number and title. We have also looped over the
review columns and the metadata columns. The second most useful technique 
is the try except block where we catch: 
IndexError – when trying to access an 
element at an index that doesn’t exist in ordered collections like lists
KeyError –  when trying to access a dictionary key that doesn't exist
TypeError – when an operation is used on an object of the wrong type
ValueError – when the correct type argument is passed to a function,
but the value is deemed inappropriate

Limitations:
The analysis uses only a 100-row metadata sample, not the full dataset.
"""

conclusions_path = Path("conclusions.txt")
conclusions_path.write_text(conclusions, encoding="utf-8")
print(f"Saved: {conclusions_path.resolve()}")
print(conclusions)
