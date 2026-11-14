# RawTree > RawTree is an analytics database for unstructured data. Send events without upfront schema setup, query them with SQL, and inspect logs from agents, scripts, and humans.

Full documentation corpus: [RawTree docs](https://rawtree.com/docs/llms.txt)

## Command line tool

Install and use the RawTree CLI from the terminal.

- [Install](https://rawtree.com/install.sh)
- [Repository](https://github.com/rawtreedb/rawtree-cli)

## Introduction

- [Introduction](https://rawtree.com/docs.md): Meet RawTree, the ClickHouse fork for raw data, and its Cloud and BYOC services.

## Get started

- [Agents](https://rawtree.com/docs/quickstart/agents.md): Give an AI agent the fastest safe path to ingest, query, and debug RawTree data.
- [CLI](https://rawtree.com/docs/quickstart/cli.md): Install rtree, create a database, ingest data, and query it.
- [API](https://rawtree.com/docs/quickstart/api.md): Ingest data into RawTree and query it through the service API.

## Concepts

- [Introduction](https://rawtree.com/docs/concepts.md): Understand how RawTree organizes data, handles evolving fields, and controls queries.
- [Resource model](https://rawtree.com/docs/concepts/resource-model.md): Understand how organizations, clusters, databases, and tables fit together.
- [Access control](https://rawtree.com/docs/concepts/access-control.md): Understand identities, credentials, database grants, row policies, and resource controls.
- [Ingestion model](https://rawtree.com/docs/concepts/ingestion-model.md): Understand table creation, evolving data, insert results, and retries.
- [Raw tables](https://rawtree.com/docs/concepts/raw-tables.md): Learn how RawTree stores and exposes JSON paths in a table.
- [Dynamic fields](https://rawtree.com/docs/concepts/dynamic-fields.md): Understand mixed types, missing values, conversion, and query semantics.
- [Queries and results](https://rawtree.com/docs/concepts/queries-and-results.md): Understand SQL scope, result metadata, query hints, and execution boundaries.
- [Storage and compute](https://rawtree.com/docs/concepts/storage-and-compute.md): Understand table parts, background work, replicas, and cluster resources.
- [Adaptive optimization](https://rawtree.com/docs/concepts/adaptive-optimization.md): Understand how RawTree adapts sorting keys, projections, and compression.
- [Sorting keys](https://rawtree.com/docs/concepts/sorting-keys.md): Understand how physical row order helps queries skip data.
- [Projections](https://rawtree.com/docs/concepts/projections.md): Understand automatic and manual projections, their lifecycle, and their costs.
- [Mutations](https://rawtree.com/docs/concepts/mutations.md): Understand background changes to existing table parts and their completion state.
- [Workload scheduling](https://rawtree.com/docs/concepts/workload-scheduling.md): Separate query workloads and share server resources deliberately.
- [Apps, connectors, and transforms](https://rawtree.com/docs/concepts/apps-connectors-and-transforms.md): Understand protocol compatibility, managed ingestion, and payload conversion.
- [SQL workflows](https://rawtree.com/docs/concepts/sql-workflows.md): Understand saved SQL, schedules, executions, and result delivery.

## Guides

- [Overview](https://rawtree.com/docs/guides.md): Follow task-focused guides for RawTree.
- [Ingest data](https://rawtree.com/docs/guides/ingest/ingest-data.md): Ingest unstructured data from inline payloads, files, and public URLs.
- [Connect Kafka](https://rawtree.com/docs/guides/ingest/kafka-connector.md): Create and manage a Kafka connector for a RawTree table.
- [Query data](https://rawtree.com/docs/guides/query/query-data.md): Read and analyze RawTree tables with SQL through the Query API.
- [Query fields with mixed types](https://rawtree.com/docs/guides/query/mixed-type-fields.md): Inspect JSON values and convert or extract their stored types.
- [Optimize queries](https://rawtree.com/docs/guides/query/optimize-queries.md): Reduce query work and compare runtime statistics.
- [Transforms](https://rawtree.com/docs/guides/transform/transforms.md): Transform your data before insert.
- [SQL workflows](https://rawtree.com/docs/guides/transform/workflows.md): Evaluate SQL on demand or on a schedule and send query results through managed sinks.
- [Restrict an API key to selected data](https://rawtree.com/docs/guides/secure/restrict-api-key-access.md): Create a database role and issue a key that can read one table.
- [Apply row-level security](https://rawtree.com/docs/guides/secure/row-level-security.md): Filter SELECT results for a database role with permissive and restrictive policies.
- [Isolate interactive and batch workloads](https://rawtree.com/docs/guides/operate/isolate-workloads.md): Build a workload tree, attach a CPU resource, and inspect scheduling.
- [Lock a role to a workload](https://rawtree.com/docs/guides/operate/settings-profiles.md): Assign a default workload to a role and prevent query overrides.
- [Change a sorting key](https://rawtree.com/docs/guides/operate/change-sorting-key.md): Set a table sorting key with ALTER TABLE MODIFY ORDER BY.
- [Use projections](https://rawtree.com/docs/guides/operate/projections.md): Add, populate, clear, and remove a table projection.
- [Manage mutations](https://rawtree.com/docs/guides/operate/mutations.md): Inspect table mutations and cancel work that is still active.
- [Manage organization members](https://rawtree.com/docs/guides/operate/manage-members.md): Invite organization members, change their roles, and remove access.
- [Monitor usage](https://rawtree.com/docs/guides/operate/monitor-usage.md): Read cluster, workflow, and query metrics through the API and SQL.

## Integrations

- [Integrations](https://rawtree.com/docs/integrations.md): Guides for using RawTree with external tools and platforms.
- [OpenTelemetry](https://rawtree.com/docs/integrations/opentelemetry.md): Send OpenTelemetry traces, logs, and metrics to RawTree with native OTLP.
- [OpenTelemetry Collector](https://rawtree.com/docs/integrations/opentelemetry-collector.md): Route OTLP telemetry through an OpenTelemetry Collector into RawTree.
- [Prometheus](https://rawtree.com/docs/integrations/prometheus.md): Store and query Prometheus metrics with service Prometheus endpoints.
- [Splunk](https://rawtree.com/docs/integrations/splunk.md): Send Splunk HEC events to RawTree and search them with SPL.
- [DataHub](https://rawtree.com/docs/integrations/datahub.md): Publish RawTree database and table metadata to your DataHub catalog.
- [Vercel Drains](https://rawtree.com/docs/integrations/vercel-drains.md): Bring Vercel logs into RawTree using Vercel Drains.
- [Daytona OpenTelemetry](https://rawtree.com/docs/integrations/daytona.md): Send Daytona sandbox telemetry to RawTree using the Daytona SDK's built-in OpenTelemetry support.
- [Supabase Postgres CDC](https://rawtree.com/docs/integrations/supabase-postgres-cdc.md): Stream Supabase Postgres changes into RawTree with logical replication and the supabase/etl worker.
- [MongoDB CDC Connector](https://rawtree.com/docs/integrations/mongodb-cdc.md): A Go-based connector that replicates data from MongoDB to RawTree using Change Streams for CDC (Change Data Capture).
- [CloudFront and WAF Security](https://rawtree.com/docs/integrations/cloudfront-waf-security.md): Stream AWS WAF and CloudFront logs into RawTree for real-time security and traffic analytics.

## Reference

- [Authentication](https://rawtree.com/docs/reference/authentication.md): Shared authentication for the API, CLI, SDK, MCP, and app protocols.
- [Overview](https://rawtree.com/docs/reference/api.md): Overview API methods, requests, and responses for RawTree Cloud and RawTree BYOC.
- [Organizations](https://rawtree.com/docs/reference/api/organizations.md): Organizations API methods, requests, and responses for RawTree Cloud and RawTree BYOC.
- [Regions](https://rawtree.com/docs/reference/api/regions.md): Regions API methods, requests, and responses for RawTree Cloud and RawTree BYOC.
- [Clusters](https://rawtree.com/docs/reference/api/clusters.md): Clusters API methods, requests, and responses for RawTree Cloud and RawTree BYOC.
- [Databases](https://rawtree.com/docs/reference/api/databases.md): Databases API methods, requests, and responses for RawTree Cloud and RawTree BYOC.
- [Tables](https://rawtree.com/docs/reference/api/tables.md): Tables API methods, requests, and responses for RawTree Cloud and RawTree BYOC.
- [Query](https://rawtree.com/docs/reference/api/query.md): Query API methods, requests, and responses for RawTree Cloud and RawTree BYOC.
- [Handlers](https://rawtree.com/docs/reference/api/handlers.md): Handlers API methods, requests, and responses for RawTree Cloud and RawTree BYOC.
- [Logs](https://rawtree.com/docs/reference/api/logs.md): Logs API methods, requests, and responses for RawTree Cloud and RawTree BYOC.
- [Metrics](https://rawtree.com/docs/reference/api/metrics.md): Metrics API methods, requests, and responses for RawTree Cloud and RawTree BYOC.
- [Connectors](https://rawtree.com/docs/reference/api/connectors.md): Connectors API methods, requests, and responses for RawTree Cloud and RawTree BYOC.
- [Workflows](https://rawtree.com/docs/reference/api/workflows.md): Workflows API methods, requests, and responses for RawTree Cloud and RawTree BYOC.
- [API keys](https://rawtree.com/docs/reference/api/keys.md): API keys API methods, requests, and responses for RawTree Cloud and RawTree BYOC.
- [Service accounts](https://rawtree.com/docs/reference/api/service-accounts.md): Service accounts API methods, requests, and responses for RawTree Cloud and RawTree BYOC.
- [Federation issuers](https://rawtree.com/docs/reference/api/federation-issuers.md): Federation issuers API methods, requests, and responses for RawTree Cloud and RawTree BYOC.
- [Federation rules](https://rawtree.com/docs/reference/api/federation-rules.md): Federation rules API methods, requests, and responses for RawTree Cloud and RawTree BYOC.
- [Overview](https://rawtree.com/docs/reference/api/apps.md): Apps API methods, requests, and responses for RawTree Cloud and RawTree BYOC.
- [OpenTelemetry](https://rawtree.com/docs/reference/api/apps/opentelemetry.md): OpenTelemetry API methods, requests, and responses for RawTree Cloud and RawTree BYOC.
- [Prometheus](https://rawtree.com/docs/reference/api/apps/prometheus.md): Prometheus API methods, requests, and responses for RawTree Cloud and RawTree BYOC.
- [Splunk](https://rawtree.com/docs/reference/api/apps/splunk.md): Splunk API methods, requests, and responses for RawTree Cloud and RawTree BYOC.
- [DataHub](https://rawtree.com/docs/reference/api/apps/datahub.md): DataHub API methods, requests, and responses for RawTree Cloud and RawTree BYOC.
- [CLI](https://rawtree.com/docs/reference/cli.md): Commands and options for the rtree CLI.
- [MCP server](https://rawtree.com/docs/reference/mcp.md): Connect your AI agent to RawTree using the hosted MCP server.
- [TypeScript SDK](https://rawtree.com/docs/reference/sdk.md): RawTree TypeScript client and OpenTelemetry helper packages.
- [Overview](https://rawtree.com/docs/reference/sql.md): SQL behavior and generated SQL reference pages.
- [SQL statements](https://rawtree.com/docs/reference/sql/statements.md): SQL statements exposed through the Query API, with syntax and permission limits.
- [SELECT](https://rawtree.com/docs/reference/sql/statements/select.md): Read data through the Query API with SELECT or WITH.
- [EXPLAIN](https://rawtree.com/docs/reference/sql/statements/explain.md): Inspect a query plan through the Query API.
- [DESCRIBE](https://rawtree.com/docs/reference/sql/statements/describe.md): Inspect columns and types with a DESCRIBE query.
- [Role statements](https://rawtree.com/docs/reference/sql/statements/role.md): CREATE ROLE and DROP ROLE support in the Query API.
- [Grant statements](https://rawtree.com/docs/reference/sql/statements/grant.md): Grant table and column privileges to a RawTree database role.
- [Row-policy statements](https://rawtree.com/docs/reference/sql/statements/row-policy.md): Create, replace, and drop row policies for database roles.
- [Settings-profile statements](https://rawtree.com/docs/reference/sql/statements/settings-profile.md): Assign workload and concurrency settings to database roles.
- [Workload and resource statements](https://rawtree.com/docs/reference/sql/statements/workload.md): Define workload trees and scheduler resources through the Query API.
- [ALTER TABLE](https://rawtree.com/docs/reference/sql/statements/alter-table.md): Supported ALTER TABLE operations in the Query API.
- [Named collections](https://rawtree.com/docs/reference/sql/statements/named-collection.md): Create named collections through the Query API.
- [KILL MUTATION](https://rawtree.com/docs/reference/sql/statements/kill-mutation.md): Stop matching active table mutations in the selected database.
- [CREATE TABLE](https://rawtree.com/docs/reference/sql/statements/create-table.md): Create a raw table with the supported engine and optional sorting key.
- [INSERT ... SELECT](https://rawtree.com/docs/reference/sql/statements/insert-select.md): Insert query results into an existing table.
- [Settings](https://rawtree.com/docs/reference/sql/settings.md): Reference for session, table, and server settings in RawTree.
- [Session settings](https://rawtree.com/docs/reference/sql/settings/session.md): Session settings available in RawTree.
- [MergeTree settings](https://rawtree.com/docs/reference/sql/settings/merge-tree.md): MergeTree settings available in RawTree.
- [Server settings](https://rawtree.com/docs/reference/sql/settings/server.md): Server settings for RawTree operators.
- [Data Types](https://rawtree.com/docs/reference/sql/data-types.md): Supported SQL data type families, aliases, and case sensitivity.
- [Formats](https://rawtree.com/docs/reference/sql/formats.md): Input and output formats supported by the RawTree SQL engine.
- [System Tables](https://rawtree.com/docs/reference/sql/system-tables.md): Comments and columns for RawTree system tables.
- [ClickHouse compatibility](https://rawtree.com/docs/reference/clickhouse-compatibility.md): Separate RawTree database behavior from Query API statement restrictions.
- [Limits](https://rawtree.com/docs/reference/limits.md): Service limits, query defaults, and rate-limit responses.
- [Errors](https://rawtree.com/docs/reference/errors.md): Error response fields, codes, statuses, and authentication metadata.

## Deployment

- [Deployment](https://rawtree.com/docs/deployment.md): Understand RawTree Cloud, RawTree BYOC, and customer-owned storage.
- [Bring your own bucket](https://rawtree.com/docs/deployment/bring-your-own-bucket.md): Verify S3 access and configure customer-owned storage for a cluster or database.

## Skills

Best practices for using RawTree from agents.

- [RawTree](https://github.com/rawtreedb/agent-skills/blob/main/skills/rawtree/SKILL.md)

## Authentication

How agents should find, use, and protect RawTree credentials.

- [Agent authentication](https://rawtree.com/auth.md)

## OpenAPI spec

The full OpenAPI specification for the RawTree API.

- [OpenAPI spec (JSON)](https://rawtree.com/openapi.json)
