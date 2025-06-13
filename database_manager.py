#!/usr/bin/env python3
"""
Advanced Database Management System
Comprehensive database operations, connection pooling, migrations, and ORM
"""

import asyncio
import time
import threading
import uuid
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Union, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, deque
from datetime import datetime, timedelta
import logging
import sqlite3
import concurrent.futures


class DatabaseType(Enum):
    """Database types"""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    MONGODB = "mongodb"
    REDIS = "redis"


class TransactionIsolation(Enum):
    """Transaction isolation levels"""
    READ_UNCOMMITTED = "READ UNCOMMITTED"
    READ_COMMITTED = "READ COMMITTED"
    REPEATABLE_READ = "REPEATABLE READ"
    SERIALIZABLE = "SERIALIZABLE"


class QueryType(Enum):
    """Query types"""
    SELECT = auto()
    INSERT = auto()
    UPDATE = auto()
    DELETE = auto()
    DDL = auto()  # Data Definition Language
    TRANSACTION = auto()


@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str = "localhost"
    port: int = 5432
    database: str = "workstation"
    username: str = "postgres"
    password: str = ""
    pool_min_size: int = 5
    pool_max_size: int = 20
    pool_timeout: float = 30.0
    query_timeout: float = 30.0
    ssl_mode: str = "prefer"
    connection_retries: int = 3
    retry_delay: float = 1.0
    auto_commit: bool = True
    isolation_level: TransactionIsolation = TransactionIsolation.READ_COMMITTED


@dataclass
class QueryResult:
    """Query execution result"""
    query_id: str
    query: str
    parameters: Optional[Dict[str, Any]]
    rows: List[Dict[str, Any]] = field(default_factory=list)
    rows_affected: int = 0
    execution_time: float = 0.0
    error: Optional[str] = None
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Migration:
    """Database migration"""
    version: str
    name: str
    up_sql: str
    down_sql: str
    description: str = ""
    dependencies: List[str] = field(default_factory=list)
    checksum: str = ""
    applied_at: Optional[datetime] = None


class DatabaseConnection:
    """Database connection wrapper"""
    
    def __init__(self, connection, config: DatabaseConfig):
        self.connection = connection
        self.config = config
        self.id = str(uuid.uuid4())
        self.created_at = time.time()
        self.last_used = time.time()
        self.in_use = False
        self.transaction_active = False
        self.query_count = 0
        self.error_count = 0
    
    async def execute(self, query: str, parameters: Optional[Dict] = None) -> QueryResult:
        """Execute query"""
        start_time = time.time()
        query_id = str(uuid.uuid4())
        
        try:
            self.last_used = time.time()
            self.query_count += 1
            
            # Execute based on database type
            if hasattr(self.connection, 'fetch'):
                # asyncpg (PostgreSQL)
                if query.strip().upper().startswith('SELECT'):
                    rows = await self.connection.fetch(query, *(parameters.values() if parameters else []))
                    result_rows = [dict(row) for row in rows]
                    rows_affected = len(result_rows)
                else:
                    result = await self.connection.execute(query, *(parameters.values() if parameters else []))
                    result_rows = []
                    rows_affected = int(result.split()[-1]) if result.startswith('UPDATE') or result.startswith('DELETE') else 1
            else:
                # Fallback for other database types
                cursor = self.connection.cursor()
                cursor.execute(query, parameters or {})
                
                if query.strip().upper().startswith('SELECT'):
                    columns = [desc[0] for desc in cursor.description]
                    result_rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
                    rows_affected = len(result_rows)
                else:
                    result_rows = []
                    rows_affected = cursor.rowcount
                
                cursor.close()
            
            execution_time = time.time() - start_time
            
            return QueryResult(
                query_id=query_id,
                query=query,
                parameters=parameters,
                rows=result_rows,
                rows_affected=rows_affected,
                execution_time=execution_time,
                success=True
            )
            
        except Exception as e:
            self.error_count += 1
            execution_time = time.time() - start_time
            
            return QueryResult(
                query_id=query_id,
                query=query,
                parameters=parameters,
                execution_time=execution_time,
                error=str(e),
                success=False
            )
    
    async def begin_transaction(self):
        """Begin transaction"""
        if hasattr(self.connection, 'transaction'):
            # asyncpg
            return self.connection.transaction()
        else:
            # Generic
            await self.execute("BEGIN")
            self.transaction_active = True
    
    async def commit(self):
        """Commit transaction"""
        if not self.transaction_active:
            return
        
        await self.execute("COMMIT")
        self.transaction_active = False
    
    async def rollback(self):
        """Rollback transaction"""
        if not self.transaction_active:
            return
        
        await self.execute("ROLLBACK")
        self.transaction_active = False
    
    async def close(self):
        """Close connection"""
        try:
            if hasattr(self.connection, 'close'):
                await self.connection.close()
            else:
                self.connection.close()
        except:
            pass
    
    def is_healthy(self) -> bool:
        """Check if connection is healthy"""
        try:
            # Simple health check
            if hasattr(self.connection, 'is_closed'):
                return not self.connection.is_closed()
            return True
        except:
            return False


class DatabasePool:
    """Advanced database connection pool"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.connections = deque()
        self.in_use_connections = {}  # connection_id -> DatabaseConnection
        self.lock = asyncio.Lock()
        
        # Pool statistics
        self.stats = {
            'total_connections': 0,
            'active_connections': 0,
            'idle_connections': 0,
            'total_queries': 0,
            'failed_queries': 0,
            'connection_errors': 0,
            'average_query_time': 0.0,
            'pool_hits': 0,
            'pool_misses': 0
        }
        
        # Health monitoring
        self.health_check_interval = 30.0
        self.health_check_task = None
        self.max_connection_age = 3600.0  # 1 hour
        self.max_idle_time = 300.0  # 5 minutes
    
    async def initialize(self) -> bool:
        """Initialize connection pool"""
        try:
            # Create minimum connections
            for _ in range(self.config.pool_min_size):
                conn = await self._create_connection()
                if conn:
                    self.connections.append(conn)
            
            # Start health monitoring
            self.health_check_task = asyncio.create_task(self._health_monitor())
            
            return len(self.connections) > 0
            
        except Exception as e:
            print(f"Pool initialization error: {e}")
            return False
    
    async def _create_connection(self) -> Optional[DatabaseConnection]:
        """Create new database connection"""
        try:
            # Create connection based on database type
            if self.config.database == "postgresql":
                try:
                    import asyncpg
                    conn = await asyncpg.connect(
                        host=self.config.host,
                        port=self.config.port,
                        database=self.config.database,
                        user=self.config.username,
                        password=self.config.password,
                        command_timeout=self.config.query_timeout,
                        server_settings={
                            'application_name': 'Innovation Workstation'
                        }
                    )
                except ImportError:
                    # Fallback to psycopg2
                    import psycopg2
                    import psycopg2.extras
                    conn = psycopg2.connect(
                        host=self.config.host,
                        port=self.config.port,
                        database=self.config.database,
                        user=self.config.username,
                        password=self.config.password
                    )
            
            elif self.config.database == "sqlite":
                import aiosqlite
                conn = await aiosqlite.connect(
                    database=self.config.database,
                    timeout=self.config.query_timeout
                )
            
            else:
                raise ValueError(f"Unsupported database type: {self.config.database}")
            
            db_conn = DatabaseConnection(conn, self.config)
            self.stats['total_connections'] += 1
            return db_conn
            
        except Exception as e:
            self.stats['connection_errors'] += 1
            print(f"Connection creation error: {e}")
            return None
    
    async def get_connection(self, timeout: Optional[float] = None) -> Optional[DatabaseConnection]:
        """Get connection from pool"""
        timeout = timeout or self.config.pool_timeout
        start_time = time.time()
        
        while (time.time() - start_time) < timeout:
            async with self.lock:
                # Try to get existing connection
                if self.connections:
                    conn = self.connections.popleft()
                    
                    # Check if connection is healthy
                    if conn.is_healthy():
                        conn.in_use = True
                        self.in_use_connections[conn.id] = conn
                        self.stats['pool_hits'] += 1
                        self.stats['active_connections'] += 1
                        self.stats['idle_connections'] = len(self.connections)
                        return conn
                    else:
                        # Connection is unhealthy, close it
                        await conn.close()
                        self.stats['total_connections'] -= 1
                
                # Create new connection if pool not at max
                if (len(self.in_use_connections) + len(self.connections)) < self.config.pool_max_size:
                    conn = await self._create_connection()
                    if conn:
                        conn.in_use = True
                        self.in_use_connections[conn.id] = conn
                        self.stats['pool_misses'] += 1
                        self.stats['active_connections'] += 1
                        return conn
            
            # Wait a bit before retrying
            await asyncio.sleep(0.1)
        
        return None
    
    async def return_connection(self, connection: DatabaseConnection):
        """Return connection to pool"""
        async with self.lock:
            if connection.id in self.in_use_connections:
                del self.in_use_connections[connection.id]
                
                # Reset connection state
                connection.in_use = False
                connection.last_used = time.time()
                
                # Rollback any active transaction
                if connection.transaction_active:
                    await connection.rollback()
                
                # Return to pool if healthy and not too old
                if (connection.is_healthy() and 
                    (time.time() - connection.created_at) < self.max_connection_age):
                    self.connections.append(connection)
                else:
                    # Close old or unhealthy connection
                    await connection.close()
                    self.stats['total_connections'] -= 1
                
                self.stats['active_connections'] = len(self.in_use_connections)
                self.stats['idle_connections'] = len(self.connections)
    
    async def _health_monitor(self):
        """Monitor pool health"""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                async with self.lock:
                    # Remove idle connections that are too old
                    current_time = time.time()
                    connections_to_remove = []
                    
                    for conn in self.connections:
                        if (current_time - conn.last_used) > self.max_idle_time:
                            connections_to_remove.append(conn)
                    
                    for conn in connections_to_remove:
                        self.connections.remove(conn)
                        await conn.close()
                        self.stats['total_connections'] -= 1
                    
                    # Ensure minimum connections
                    while len(self.connections) < self.config.pool_min_size:
                        conn = await self._create_connection()
                        if conn:
                            self.connections.append(conn)
                        else:
                            break
                    
                    self.stats['idle_connections'] = len(self.connections)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Health monitor error: {e}")
    
    async def close_all(self):
        """Close all connections"""
        async with self.lock:
            # Close idle connections
            while self.connections:
                conn = self.connections.popleft()
                await conn.close()
            
            # Close active connections
            for conn in self.in_use_connections.values():
                await conn.close()
            
            self.in_use_connections.clear()
            self.stats['total_connections'] = 0
            self.stats['active_connections'] = 0
            self.stats['idle_connections'] = 0
        
        # Stop health monitor
        if self.health_check_task:
            self.health_check_task.cancel()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        return self.stats.copy()


class MigrationManager:
    """Database migration management"""
    
    def __init__(self, database_manager):
        self.db_manager = database_manager
        self.migrations = {}  # version -> Migration
        self.migrations_table = "schema_migrations"
        self.applied_migrations = set()
    
    async def setup(self):
        """Setup migration system"""
        await self._create_migrations_table()
        await self._load_applied_migrations()
    
    async def _create_migrations_table(self):
        """Create migrations tracking table"""
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.migrations_table} (
            version VARCHAR(255) PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            checksum VARCHAR(64) NOT NULL
        )
        """
        
        await self.db_manager.execute(create_table_sql)
    
    async def _load_applied_migrations(self):
        """Load applied migrations from database"""
        query = f"SELECT version FROM {self.migrations_table}"
        result = await self.db_manager.execute(query)
        
        if result.success:
            self.applied_migrations = {row['version'] for row in result.rows}
    
    def add_migration(self, migration: Migration):
        """Add migration"""
        # Calculate checksum
        migration.checksum = hashlib.sha256(
            (migration.up_sql + migration.down_sql).encode()
        ).hexdigest()
        
        self.migrations[migration.version] = migration
    
    async def migrate_up(self, target_version: str = None) -> bool:
        """Run migrations up to target version"""
        try:
            # Get migrations to apply
            migrations_to_apply = []
            
            for version in sorted(self.migrations.keys()):
                if version not in self.applied_migrations:
                    migrations_to_apply.append(self.migrations[version])
                    
                    if target_version and version == target_version:
                        break
            
            # Apply migrations
            for migration in migrations_to_apply:
                success = await self._apply_migration(migration)
                if not success:
                    return False
            
            return True
            
        except Exception as e:
            print(f"Migration error: {e}")
            return False
    
    async def migrate_down(self, target_version: str) -> bool:
        """Rollback migrations down to target version"""
        try:
            # Get migrations to rollback
            migrations_to_rollback = []
            
            for version in sorted(self.migrations.keys(), reverse=True):
                if version in self.applied_migrations:
                    if version == target_version:
                        break
                    migrations_to_rollback.append(self.migrations[version])
            
            # Rollback migrations
            for migration in migrations_to_rollback:
                success = await self._rollback_migration(migration)
                if not success:
                    return False
            
            return True
            
        except Exception as e:
            print(f"Migration rollback error: {e}")
            return False
    
    async def _apply_migration(self, migration: Migration) -> bool:
        """Apply single migration"""
        try:
            # Execute migration SQL
            result = await self.db_manager.execute(migration.up_sql)
            
            if not result.success:
                print(f"Migration {migration.version} failed: {result.error}")
                return False
            
            # Record migration
            record_sql = f"""
            INSERT INTO {self.migrations_table} (version, name, checksum)
            VALUES (%(version)s, %(name)s, %(checksum)s)
            """
            
            record_result = await self.db_manager.execute(record_sql, {
                'version': migration.version,
                'name': migration.name,
                'checksum': migration.checksum
            })
            
            if record_result.success:
                self.applied_migrations.add(migration.version)
                print(f"Migration {migration.version} applied successfully")
                return True
            else:
                print(f"Failed to record migration {migration.version}")
                return False
            
        except Exception as e:
            print(f"Migration application error: {e}")
            return False
    
    async def _rollback_migration(self, migration: Migration) -> bool:
        """Rollback single migration"""
        try:
            # Execute rollback SQL
            result = await self.db_manager.execute(migration.down_sql)
            
            if not result.success:
                print(f"Migration rollback {migration.version} failed: {result.error}")
                return False
            
            # Remove migration record
            remove_sql = f"DELETE FROM {self.migrations_table} WHERE version = %(version)s"
            remove_result = await self.db_manager.execute(remove_sql, {
                'version': migration.version
            })
            
            if remove_result.success:
                self.applied_migrations.discard(migration.version)
                print(f"Migration {migration.version} rolled back successfully")
                return True
            else:
                print(f"Failed to remove migration record {migration.version}")
                return False
            
        except Exception as e:
            print(f"Migration rollback error: {e}")
            return False


class PostgreSQLManager:
    """PostgreSQL database manager"""
    
    def __init__(self, app):
        self.app = app
        self.config = None
        self.pool = None
        self.migration_manager = None
        
        # Query statistics
        self.query_stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'total_execution_time': 0.0,
            'average_execution_time': 0.0,
            'queries_by_type': defaultdict(int),
            'slow_queries': deque(maxlen=100)  # Track slow queries
        }
        
        # Connection monitoring
        self.connection_timeout = 30.0
        self.query_timeout = 30.0
        self.slow_query_threshold = 1.0  # 1 second
        
        self.lock = asyncio.Lock()
    
    async def setup(self) -> bool:
        """Setup PostgreSQL manager"""
        try:
            # Load configuration
            self.config = DatabaseConfig(
                host=await self.app.config.get('database.host', 'localhost'),
                port=await self.app.config.get('database.port', 5432),
                database=await self.app.config.get('database.name', 'workstation'),
                username=await self.app.config.get('database.username', 'postgres'),
                password=await self.app.config.get('database.password', ''),
                pool_min_size=await self.app.config.get('database.pool_min_size', 5),
                pool_max_size=await self.app.config.get('database.pool_max_size', 20),
                pool_timeout=await self.app.config.get('database.pool_timeout', 30.0),
                query_timeout=await self.app.config.get('database.query_timeout', 30.0)
            )
            
            # Initialize connection pool
            self.pool = DatabasePool(self.config)
            pool_initialized = await self.pool.initialize()
            
            if not pool_initialized:
                self.app.logger.warning("Database pool initialization failed")
                return False
            
            # Initialize migration manager
            self.migration_manager = MigrationManager(self)
            await self.migration_manager.setup()
            
            # Load default migrations
            await self._load_default_migrations()
            
            self.app.logger.info("PostgreSQL manager initialized successfully")
            return True
            
        except Exception as e:
            self.app.error_handler.handle_error(e, "PostgreSQLManager.setup")
            return False
    
    async def _load_default_migrations(self):
        """Load default system migrations"""
        # Example migration for audit log table
        audit_migration = Migration(
            version="001",
            name="create_audit_log",
            up_sql="""
            CREATE TABLE IF NOT EXISTS audit_log (
                id SERIAL PRIMARY KEY,
                table_name VARCHAR(255) NOT NULL,
                operation VARCHAR(10) NOT NULL,
                record_id VARCHAR(255),
                old_values JSONB,
                new_values JSONB,
                user_id VARCHAR(255),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON audit_log(timestamp);
            CREATE INDEX IF NOT EXISTS idx_audit_log_table ON audit_log(table_name);
            """,
            down_sql="DROP TABLE IF EXISTS audit_log;",
            description="Create audit log table for tracking data changes"
        )
        
        self.migration_manager.add_migration(audit_migration)
        
        # Run initial migrations
        await self.migration_manager.migrate_up()
    
    async def execute(self, query: str, parameters: Optional[Dict] = None) -> QueryResult:
        """Execute database query"""
        start_time = time.time()
        
        try:
            # Get connection from pool
            connection = await self.pool.get_connection()
            if not connection:
                raise RuntimeError("Could not get database connection")
            
            try:
                # Execute query
                result = await connection.execute(query, parameters)
                
                # Update statistics
                execution_time = time.time() - start_time
                self._update_query_stats(query, execution_time, result.success)
                
                # Track slow queries
                if execution_time > self.slow_query_threshold:
                    self.query_stats['slow_queries'].append({
                        'query': query[:500],  # Truncate long queries
                        'execution_time': execution_time,
                        'timestamp': time.time()
                    })
                
                return result
                
            finally:
                # Return connection to pool
                await self.pool.return_connection(connection)
        
        except Exception as e:
            execution_time = time.time() - start_time
            self._update_query_stats(query, execution_time, False)
            
            return QueryResult(
                query_id=str(uuid.uuid4()),
                query=query,
                parameters=parameters,
                execution_time=execution_time,
                error=str(e),
                success=False
            )
    
    def _update_query_stats(self, query: str, execution_time: float, success: bool):
        """Update query statistics"""
        self.query_stats['total_queries'] += 1
        self.query_stats['total_execution_time'] += execution_time
        
        if success:
            self.query_stats['successful_queries'] += 1
        else:
            self.query_stats['failed_queries'] += 1
        
        # Update average
        self.query_stats['average_execution_time'] = (
            self.query_stats['total_execution_time'] / 
            self.query_stats['total_queries']
        )
        
        # Update query type stats
        query_type = query.strip().split()[0].upper()
        self.query_stats['queries_by_type'][query_type] += 1
    
    async def transaction(self):
        """Get transaction context"""
        connection = await self.pool.get_connection()
        if not connection:
            raise RuntimeError("Could not get database connection")
        
        return DatabaseTransaction(self, connection)
    
    async def close_async(self):
        """Close database manager"""
        if self.pool:
            await self.pool.close_all()
        
        self.app.logger.info("PostgreSQL manager closed")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        pool_stats = self.pool.get_stats() if self.pool else {}
        
        return {
            'query_stats': self.query_stats,
            'pool_stats': pool_stats,
            'config': {
                'host': self.config.host if self.config else None,
                'port': self.config.port if self.config else None,
                'database': self.config.database if self.config else None,
                'pool_min_size': self.config.pool_min_size if self.config else None,
                'pool_max_size': self.config.pool_max_size if self.config else None
            }
        }


class DatabaseTransaction:
    """Database transaction context manager"""
    
    def __init__(self, db_manager, connection):
        self.db_manager = db_manager
        self.connection = connection
        self.transaction = None
        self.completed = False
    
    async def __aenter__(self):
        self.transaction = await self.connection.begin_transaction()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            if exc_type is None and not self.completed:
                await self.commit()
            else:
                await self.rollback()
        finally:
            await self.db_manager.pool.return_connection(self.connection)
    
    async def execute(self, query: str, parameters: Optional[Dict] = None) -> QueryResult:
        """Execute query within transaction"""
        return await self.connection.execute(query, parameters)
    
    async def commit(self):
        """Commit transaction"""
        if self.transaction and hasattr(self.transaction, '__aexit__'):
            await self.transaction.__aexit__(None, None, None)
        else:
            await self.connection.commit()
        self.completed = True
    
    async def rollback(self):
        """Rollback transaction"""
        if self.transaction and hasattr(self.transaction, '__aexit__'):
            await self.transaction.__aexit__(Exception, Exception(), None)
        else:
            await self.connection.rollback()
        self.completed = True