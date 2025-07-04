"""
Spark ETL Pipeline for MIMIC-IV Data Processing
Processes waveform and hospital data to create training-ready datasets.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pyarrow as pa
import yaml
from delta import configure_spark_with_delta_pip
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MIMICETLPipeline:
    """ETL Pipeline for MIMIC-IV data processing."""
    
    def __init__(self, config_path: str, spark_config: Optional[Dict] = None):
        """Initialize the ETL pipeline.
        
        Args:
            config_path: Path to the configuration YAML file
            spark_config: Optional Spark configuration overrides
        """
        self.config = self._load_config(config_path)
        self.spark = self._create_spark_session(spark_config)
        self.stats = {}
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _create_spark_session(self, spark_config: Optional[Dict] = None) -> SparkSession:
        """Create Spark session with Delta Lake support."""
        builder = (SparkSession.builder
                  .appName("MIMIC-IV-ETL")
                  .config("spark.sql.adaptive.enabled", "true")
                  .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
                  .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
                  .config("spark.sql.execution.arrow.pyspark.enabled", "true"))
        
        if spark_config:
            for key, value in spark_config.items():
                builder = builder.config(key, value)
        
        spark = configure_spark_with_delta_pip(builder).getOrCreate()
        spark.sparkContext.setLogLevel("WARN")
        return spark
    
    def extract_waveforms(self, input_path: str) -> DataFrame:
        """Extract and preprocess waveform data.
        
        Args:
            input_path: Path to MIMIC-IV waveform files
            
        Returns:
            Preprocessed waveform DataFrame
        """
        logger.info("Extracting waveform data...")
        
        # Define waveform schema
        waveform_schema = StructType([
            StructField("subject_id", IntegerType(), True),
            StructField("stay_id", IntegerType(), True),
            StructField("charttime", TimestampType(), True),
            StructField("signal_name", StringType(), True),
            StructField("signal_value", DoubleType(), True),
            StructField("fs", IntegerType(), True)  # sampling frequency
        ])
        
        # Read waveform files
        df = self.spark.read.schema(waveform_schema).parquet(input_path)
        
        # Filter to 1Hz sampling and relevant signals
        target_signals = self.config['waveform']['signals']
        df = df.filter(
            (F.col("fs") == 1) &
            (F.col("signal_name").isin(target_signals))
        )
        
        # Create 30-minute windows
        df = df.withColumn(
            "window_start",
            F.date_trunc("hour", F.col("charttime")) + 
            F.expr("INTERVAL 30 MINUTES") * 
            F.floor(F.minute(F.col("charttime")) / 30)
        )
        
        # Pivot signals to columns
        df = df.groupBy("subject_id", "stay_id", "window_start").pivot("signal_name").agg(
            F.avg("signal_value").alias("mean"),
            F.stddev("signal_value").alias("std"),
            F.min("signal_value").alias("min"),
            F.max("signal_value").alias("max"),
            F.count("signal_value").alias("count")
        )
        
        return df
    
    def extract_ehr_data(self, input_path: str) -> DataFrame:
        """Extract and preprocess EHR tabular data.
        
        Args:
            input_path: Path to MIMIC-IV hospital data
            
        Returns:
            Preprocessed EHR DataFrame
        """
        logger.info("Extracting EHR data...")
        
        # Read core tables
        admissions = self.spark.read.csv(
            f"{input_path}/hosp/admissions.csv.gz", 
            header=True, inferSchema=True
        )
        
        patients = self.spark.read.csv(
            f"{input_path}/hosp/patients.csv.gz",
            header=True, inferSchema=True
        )
        
        icustays = self.spark.read.csv(
            f"{input_path}/icu/icustays.csv.gz",
            header=True, inferSchema=True
        )
        
        # Lab events
        labevents = self.spark.read.csv(
            f"{input_path}/hosp/labevents.csv.gz",
            header=True, inferSchema=True
        )
        
        # Chart events (vitals)
        chartevents = self.spark.read.csv(
            f"{input_path}/icu/chartevents.csv.gz",
            header=True, inferSchema=True
        )
        
        # Join core demographic data
        ehr_df = (icustays
                 .join(admissions, "hadm_id")
                 .join(patients, "subject_id")
                 .select(
                     "subject_id", "stay_id", "hadm_id",
                     "intime", "outtime",
                     "gender", "anchor_age", "race",
                     "admission_type", "admission_location"
                 ))
        
        # Process lab events
        target_labs = self.config['ehr']['lab_items']
        labs_df = (labevents
                  .filter(F.col("itemid").isin(target_labs))
                  .withColumn("charttime_hour", F.date_trunc("hour", F.col("charttime")))
                  .groupBy("subject_id", "charttime_hour", "itemid")
                  .agg(F.avg("valuenum").alias("lab_value"))
                  .groupBy("subject_id", "charttime_hour")
                  .pivot("itemid")
                  .agg(F.first("lab_value")))
        
        # Process vital signs
        target_vitals = self.config['ehr']['vital_items']
        vitals_df = (chartevents
                    .filter(F.col("itemid").isin(target_vitals))
                    .withColumn("charttime_hour", F.date_trunc("hour", F.col("charttime")))
                    .groupBy("subject_id", "charttime_hour", "itemid")
                    .agg(F.avg("valuenum").alias("vital_value"))
                    .groupBy("subject_id", "charttime_hour")
                    .pivot("itemid")
                    .agg(F.first("vital_value")))
        
        # Join all EHR data
        ehr_df = (ehr_df
                 .join(labs_df, "subject_id", "left")
                 .join(vitals_df, "subject_id", "left"))
        
        return ehr_df
    
    def create_sepsis_labels(self, ehr_df: DataFrame) -> DataFrame:
        """Create Sepsis-3 labels with 6-hour prediction window.
        
        Args:
            ehr_df: EHR DataFrame with patient data
            
        Returns:
            DataFrame with sepsis labels
        """
        logger.info("Creating Sepsis-3 labels...")
        
        # Sepsis-3 criteria: SOFA >= 2 + suspected infection
        # Simplified implementation - in practice would need more complex logic
        
        # Calculate SOFA components (simplified)
        df_with_sofa = ehr_df.withColumn(
            "sofa_score",
            # Respiratory (PaO2/FiO2 ratio)
            F.when(F.col("50821") < 400, 1)  # PaO2 item_id example
            .when(F.col("50821") < 300, 2)
            .when(F.col("50821") < 200, 3)
            .when(F.col("50821") < 100, 4)
            .otherwise(0) +
            # Cardiovascular (MAP, vasopressors)
            F.when(F.col("220052") < 70, 1)  # MAP item_id example
            .otherwise(0) +
            # Hepatic (bilirubin)
            F.when(F.col("50885") > 1.2, 1)  # Bilirubin item_id example
            .when(F.col("50885") > 2.0, 2)
            .when(F.col("50885") > 6.0, 3)
            .when(F.col("50885") > 12.0, 4)
            .otherwise(0) +
            # Coagulation (platelets)
            F.when(F.col("51265") < 150, 1)  # Platelets item_id example
            .when(F.col("51265") < 100, 2)
            .when(F.col("51265") < 50, 3)
            .when(F.col("51265") < 20, 4)
            .otherwise(0) +
            # Neurological (GCS - would need separate calculation)
            F.lit(0) +  # Placeholder
            # Renal (creatinine, urine output)
            F.when(F.col("50912") > 1.2, 1)  # Creatinine item_id example
            .when(F.col("50912") > 2.0, 2)
            .when(F.col("50912") > 3.5, 3)
            .when(F.col("50912") > 5.0, 4)
            .otherwise(0)
        )
        
        # Define sepsis onset (SOFA >= 2)
        window_spec = Window.partitionBy("subject_id", "stay_id").orderBy("charttime_hour")
        
        labels_df = df_with_sofa.withColumn(
            "sepsis_onset",
            F.when(F.col("sofa_score") >= 2, F.col("charttime_hour")).otherwise(None)
        ).withColumn(
            "sepsis_onset_time",
            F.first("sepsis_onset", ignorenulls=True).over(window_spec)
        ).withColumn(
            "hours_to_sepsis",
            (F.col("sepsis_onset_time").cast("long") - F.col("charttime_hour").cast("long")) / 3600
        ).withColumn(
            "sepsis_label",
            F.when(
                (F.col("hours_to_sepsis") >= 0) & (F.col("hours_to_sepsis") <= 6), 1
            ).otherwise(0)
        )
        
        return labels_df
    
    def normalize_features(self, df: DataFrame, feature_cols: List[str]) -> Tuple[DataFrame, Dict]:
        """Normalize numeric features and save statistics.
        
        Args:
            df: Input DataFrame
            feature_cols: List of feature columns to normalize
            
        Returns:
            Tuple of (normalized DataFrame, normalization statistics)
        """
        logger.info("Normalizing features...")
        
        stats = {}
        normalized_df = df
        
        for col in feature_cols:
            if col in df.columns:
                # Calculate min-max statistics
                min_max = df.select(
                    F.min(col).alias("min"),
                    F.max(col).alias("max")
                ).collect()[0]
                
                min_val, max_val = min_max["min"], min_max["max"]
                
                if min_val is not None and max_val is not None and max_val != min_val:
                    stats[col] = {"min": min_val, "max": max_val}
                    
                    # Apply min-max normalization
                    normalized_df = normalized_df.withColumn(
                        f"{col}_norm",
                        (F.col(col) - min_val) / (max_val - min_val)
                    )
                else:
                    logger.warning(f"Skipping normalization for column {col} due to invalid range")
        
        return normalized_df, stats
    
    def create_sliding_windows(self, df: DataFrame, window_size: int = 72) -> DataFrame:
        """Create sliding windows for sequence prediction.
        
        Args:
            df: Input DataFrame with time series data
            window_size: Number of time steps in each window (default: 72 = 36 hours)
            
        Returns:
            DataFrame with sliding windows
        """
        logger.info(f"Creating sliding windows of size {window_size}...")
        
        # Create row numbers within each stay
        window_spec = Window.partitionBy("subject_id", "stay_id").orderBy("charttime_hour")
        
        df_windowed = df.withColumn("row_num", F.row_number().over(window_spec))
        
        # Create windows using array collection
        window_data = []
        for i in range(window_size):
            window_data.append(
                F.lag(F.col("row_num"), i).over(window_spec).alias(f"window_{i}")
            )
        
        df_with_windows = df_windowed.select(
            "subject_id", "stay_id", "charttime_hour", "sepsis_label",
            *window_data
        ).filter(F.col(f"window_{window_size-1}").isNotNull())
        
        return df_with_windows
    
    def save_to_delta(self, df: DataFrame, output_path: str, partition_cols: List[str] = None):
        """Save DataFrame to Delta Lake format.
        
        Args:
            df: DataFrame to save
            output_path: Output path for Delta table
            partition_cols: Optional partition columns
        """
        logger.info(f"Saving to Delta Lake: {output_path}")
        
        writer = df.write.format("delta").mode("overwrite")
        
        if partition_cols:
            writer = writer.partitionBy(*partition_cols)
        
        writer.save(output_path)
    
    def save_stats(self, output_path: str):
        """Save normalization statistics to JSON."""
        import json
        with open(f"{output_path}/stats.json", "w") as f:
            json.dump(self.stats, f, indent=2)
    
    def run_pipeline(self, 
                    waveform_path: str,
                    ehr_path: str, 
                    output_path: str):
        """Run the complete ETL pipeline.
        
        Args:
            waveform_path: Path to MIMIC-IV waveform data
            ehr_path: Path to MIMIC-IV hospital data
            output_path: Output path for processed data
        """
        logger.info("Starting MIMIC-IV ETL pipeline...")
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Extract data
        waveform_df = self.extract_waveforms(waveform_path)
        ehr_df = self.extract_ehr_data(ehr_path)
        
        # Create labels
        labeled_df = self.create_sepsis_labels(ehr_df)
        
        # Join waveform and EHR data
        # Convert window_start to match charttime_hour for joining
        waveform_df = waveform_df.withColumnRenamed("window_start", "charttime_hour")
        
        combined_df = labeled_df.join(
            waveform_df, 
            ["subject_id", "stay_id", "charttime_hour"], 
            "inner"
        )
        
        # Get feature columns for normalization
        feature_cols = [col for col in combined_df.columns 
                       if col not in ["subject_id", "stay_id", "charttime_hour", 
                                    "sepsis_label", "sepsis_onset_time", "hours_to_sepsis"]]
        
        # Normalize features
        normalized_df, self.stats = self.normalize_features(combined_df, feature_cols)
        
        # Create sliding windows
        windowed_df = self.create_sliding_windows(normalized_df)
        
        # Save processed data
        self.save_to_delta(
            windowed_df, 
            f"{output_path}/curated/training_data",
            partition_cols=["subject_id"]
        )
        
        # Save labels manifest
        labels_df = windowed_df.select(
            "subject_id", "stay_id", "charttime_hour", "sepsis_label"
        )
        labels_df.coalesce(1).write.mode("overwrite").csv(
            f"{output_path}/labels.csv", header=True
        )
        
        # Save normalization statistics
        self.save_stats(output_path)
        
        logger.info("ETL pipeline completed successfully!")
    
    def stop(self):
        """Stop Spark session."""
        self.spark.stop()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MIMIC-IV ETL Pipeline")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--waveform-path", required=True, help="Path to waveform data")
    parser.add_argument("--ehr-path", required=True, help="Path to EHR data")
    parser.add_argument("--output-path", required=True, help="Output path for processed data")
    
    args = parser.parse_args()
    
    # Run pipeline
    pipeline = MIMICETLPipeline(args.config)
    try:
        pipeline.run_pipeline(
            args.waveform_path,
            args.ehr_path,
            args.output_path
        )
    finally:
        pipeline.stop()
