# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# dependencies = [
#   "kafka-python",
# ]
# ///
# MAGIC %pip install kafka-python

# COMMAND ----------

# MAGIC %run ./00_ingest_config

# COMMAND ----------

from kafka.admin import KafkaAdminClient, NewTopic

# COMMAND ----------

kafka_bootstrap_servers = dbutils.secrets.get(KAFKA_SECRET_SCOPE, KAFKA_SECRET_KEY)

# Same brokers and protocol the streaming notebook uses.
security_protocol = "PLAINTEXT"
num_partitions = 8

# COMMAND ----------

def delete_topic(topic_name):
    admin_client = KafkaAdminClient(
        bootstrap_servers=kafka_bootstrap_servers.split(','),
        security_protocol=security_protocol,
        client_id="delete-topic-client"
    )
    try:
        admin_client.delete_topics(topics=[topic_name])
        print(f"Topic '{topic_name}' marked for deletion.")
    except Exception as e:
        print(f"Failed to delete topic '{topic_name}': {e}")
    finally:
        admin_client.close()

delete_topic(TXN_TOPIC)

# COMMAND ----------

retention_ms = -1

def create_topic(topic_name, num_partitions):
    create_client = KafkaAdminClient(
        bootstrap_servers=kafka_bootstrap_servers.split(','),
        security_protocol=security_protocol,
        client_id="create-topic-client"
    )

    topic = NewTopic(
        name=topic_name,
        num_partitions=num_partitions,
        replication_factor=3,
        topic_configs={
            'retention.ms': str(retention_ms)
        }
    )

    try:
        create_client.create_topics(new_topics=[topic], validate_only=False)
        print(f"Topic '{topic_name}' created successfully.")
    except Exception as e:
        print(f"Error creating topic: {e}")
    finally:
        create_client.close()

create_topic(TXN_TOPIC, num_partitions)
