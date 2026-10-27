
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    sum as spark_sum,
    count,
    date_format,
    lag,
    round as spark_round
)
from pyspark.sql.window import Window

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"

spark = (
    SparkSession.builder
    .appName("MonthlySalesAnalytics")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    # 1. Read orders
    orders = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(str(DATA_DIR / "orders.csv"))
    )

    # 2. Basic cleaning
    orders_clean = (
        orders.dropDuplicates(["order_id"])
        .dropna(subset=[
            "order_id", "date", "quantity", "total", "status"
        ])
        .filter(
            (col("quantity") > 0) &
            (col("total") >= 0)
        )
    )

    # 3. Create month column
    orders_clean = orders_clean.withColumn(
        "month",
        date_format(col("date"), "yyyy-MM")
    )

    # 4. Monthly sales report
    monthly_sales = (
        orders_clean.groupBy("month")
        .agg(
            spark_round(spark_sum("total"), 2)
            .alias("monthly_revenue"),
            count("order_id").alias("total_orders")
        )
        .orderBy("month")
    )

    print("\n===== MONTHLY SALES =====")
    monthly_sales.show(truncate=False)

    # 5. Month-over-month revenue comparison
    month_window = Window.orderBy("month")

    monthly_comparison = (
        monthly_sales
        .withColumn(
            "previous_month_revenue",
            lag("monthly_revenue").over(month_window)
        )
        .withColumn(
            "revenue_change",
            spark_round(
                col("monthly_revenue") -
                col("previous_month_revenue"),
                2
            )
        )
        .withColumn(
            "growth_percent",
            spark_round(
                (
                    col("monthly_revenue") -
                    col("previous_month_revenue")
                ) / col("previous_month_revenue") * 100,
                2
            )
        )
    )

    print("\n===== MONTH-OVER-MONTH ANALYSIS =====")
    monthly_comparison.show(truncate=False)

    # 6. Revenue by order status
    status_report = (
        orders_clean.groupBy("status")
        .agg(
            spark_round(spark_sum("total"), 2)
            .alias("total_revenue"),
            count("order_id").alias("total_orders")
        )
        .orderBy(col("total_revenue").desc())
    )

    print("\n===== STATUS REPORT =====")
    status_report.show(truncate=False)

    # 7. Save reports
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    monthly_sales.write.mode("overwrite") \
        .option("header", True) \
        .csv(str(OUTPUT_DIR / "monthly_sales"))

    monthly_comparison.write.mode("overwrite") \
        .option("header", True) \
        .csv(str(OUTPUT_DIR / "monthly_comparison"))

    status_report.write.mode("overwrite") \
        .option("header", True) \
        .csv(str(OUTPUT_DIR / "status_report"))

    print("\nReports saved successfully in output folder.")

finally:
    spark.stop()
