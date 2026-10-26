# Transformational AI — Exploring Amazon Electronics

A small Python data analysis project exploring the prices and ratings of highly rated electronics on Amazon.

## The data

The script streams the first 100 electronics reviews and 100 product records from the [Amazon Reviews 2023 dataset](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023). Reviews are inspected separately; the price and rating analysis uses the product records.

Products are kept when they have a nonempty title, a numeric price, and an average rating of at least 4.5 out of 5. Invalid price and rating values are converted to missing values before filtering. The cleaned records are saved as `top_products.csv` and `top_products.parquet`.

## How the data is displayed

Tables and printed summaries show sample records, column names, data types, and product titles. Seaborn and Matplotlib charts show:

- A price histogram with a smooth distribution curve; its visible bars cycle through yellow, blue, and red.
- Scatterplots comparing price with average rating, including a closer look at products rated 4.7 or higher.
- Side-by-side price and rating charts, plus a table sorted by highest rating and then lowest price.

## Results in the saved sample

Of the 100 product records, **21 meet the filters**. Of these, **12 (57.1%) have ratings of at least 4.7**, and four have a perfect 5.0 rating.

Prices range from **3.99 to 909.99**, with a median of **24.63**. Eighteen products cost under 100, two cost between 100 and 200, and one costs over 800. Highly rated products appear at several price levels, with most concentrated at the lower end.

These findings describe the first 100 records, not a random sample or the full electronics market.

## Python techniques used

– ** for loop – 10 loops on attributes such as the products' number and title, review columns, and the metadata columns 
- **Streaming and `islice`** to limit how much data is loaded into memory.
- **pandas DataFrames** for numeric conversion, missing-value checks, filtering, copying, and sorting.
- **Functions and `try`/`except`** for reusable calculations and safe product lookups, including handling invalid indexes and values.
- **Loops, lists, `enumerate`, and a generator expression** to list products and color nonempty histogram bars.
- **f-strings and `pathlib`** to format summaries and save files, alongside CSV and Parquet exports.
