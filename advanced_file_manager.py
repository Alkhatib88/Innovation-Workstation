#!/usr/bin/env python3
"""
Advanced File and Directory Management Systems
Comprehensive file operations, monitoring, versioning, and security
"""

import asyncio
import os
import shutil
import hashlib
import mimetypes
import time
import json
import pickle
import gzip
import tarfile
import zipfile
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, deque
import stat
import fnmatch
import tempfile
import uuid
from datetime import datetime, timedelta
import watchdog
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class FileType(Enum):
    """File type classification"""
    TEXT = auto()
    BINARY = auto()
    IMAGE = auto()
    VIDEO = auto()
    AUDIO = auto()
    DOCUMENT = auto()
    ARCHIVE = auto()
    EXECUTABLE = auto()
    CONFIG = auto()
    LOG = auto()
    DATABASE = auto()
    TEMPORARY = auto()
    UNKNOWN = auto()


class FileOperation(Enum):
    """File operations"""
    CREATE = auto()
    READ = auto()
    WRITE = auto()
    UPDATE = auto()
    DELETE = auto()
    MOVE = auto()
    COPY = auto()
    RENAME = auto()
    COMPRESS = auto()
    DECOMPRESS = auto()
    ENCRYPT = auto()
    DECRYPT = auto()


class FilePermission(Enum):
    """File permissions"""
    READ = 'r'
    WRITE = 'w'
    EXECUTE = 'x'
    READ_WRITE = 'rw'
    READ_EXECUTE = 'rx'
    WRITE_EXECUTE = 'wx'
    ALL = 'rwx'


@dataclass
class FileMetadata:
    """Enhanced file metadata"""
    path: Path
    size: int = 0
    created_time: float = 0
    modified_time: float = 0
    accessed_time: float = 0
    file_type: FileType = FileType.UNKNOWN
    mime_type: str = ""
    checksum: str = ""
    permissions: str = ""
    owner: str = ""
    group: str = ""
    is_symlink: bool = False
    target_path: Optional[Path] = None
    encoding: Optional[str] = None
    line_count: Optional[int] = None
    version: int = 1
    tags: Set[str] = field(default_factory=set)
    custom_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FileVersion:
    """File version information"""
    version: int
    timestamp: float
    size: int
    checksum: str
    author: str = ""
    comment: str = ""
    backup_path: Optional[Path] = None


class FileIndex:
    """File indexing and search system"""
    
    def __init__(self, index_path: Path):
        self.index_path = index_path
        self.index = {}  # path -> FileMetadata
        self.checksum_index = {}  # checksum -> List[path]
        self.type_index = defaultdict(list)  # file_type -> List[path]
        self.tag_index = defaultdict(list)  # tag -> List[path]
        self.size_index = defaultdict(list)  # size_range -> List[path]
        self.lock = threading.Lock()
        
        self.load_index()
    
    def add_file(self, metadata: FileMetadata):
        """Add file to index"""
        with self.lock:
            path_str = str(metadata.path)
            self.index[path_str] = metadata
            
            # Update secondary indexes
            if metadata.checksum:
                if metadata.checksum not in self.checksum_index:
                    self.checksum_index[metadata.checksum] = []
                self.checksum_index[metadata.checksum].append(path_str)
            
            self.type_index[metadata.file_type].append(path_str)
            
            for tag in metadata.tags:
                self.tag_index[tag].append(path_str)
            
            # Size index (categorize by size ranges)
            size_range = self._get_size_range(metadata.size)
            self.size_index[size_range].append(path_str)
    
    def remove_file(self, path: Path):
        """Remove file from index"""
        with self.lock:
            path_str = str(path)
            if path_str not in self.index:
                return
            
            metadata = self.index[path_str]
            
            # Remove from secondary indexes
            if metadata.checksum and metadata.checksum in self.checksum_index:
                if path_str in self.checksum_index[metadata.checksum]:
                    self.checksum_index[metadata.checksum].remove(path_str)
                if not self.checksum_index[metadata.checksum]:
                    del self.checksum_index[metadata.checksum]
            
            if path_str in self.type_index[metadata.file_type]:
                self.type_index[metadata.file_type].remove(path_str)
            
            for tag in metadata.tags:
                if path_str in self.tag_index[tag]:
                    self.tag_index[tag].remove(path_str)
            
            size_range = self._get_size_range(metadata.size)
            if path_str in self.size_index[size_range]:
                self.size_index[size_range].remove(path_str)
            
            del self.index[path_str]
    
    def search_files(self, query: str = "", file_type: FileType = None,
                    tags: List[str] = None, size_range: Tuple[int, int] = None,
                    modified_after: float = None) -> List[FileMetadata]:
        """Search files with various criteria"""
        with self.lock:
            results = []
            
            # Start with all files if no specific criteria
            candidates = set(self.index.keys())
            
            # Filter by file type
            if file_type:
                type_files = set(self.type_index.get(file_type, []))
                candidates &= type_files
            
            # Filter by tags
            if tags:
                for tag in tags:
                    tag_files = set(self.tag_index.get(tag, []))
                    candidates &= tag_files
            
            # Filter by size range
            if size_range:
                min_size, max_size = size_range
                size_files = set()
                for size_cat, files in self.size_index.items():
                    if size_cat[0] <= max_size and size_cat[1] >= min_size:
                        size_files.update(files)
                candidates &= size_files
            
            # Filter by modification time
            if modified_after:
                time_candidates = set()
                for path_str in candidates:
                    metadata = self.index[path_str]
                    if metadata.modified_time >= modified_after:
                        time_candidates.add(path_str)
                candidates = time_candidates
            
            # Text search in filename
            if query:
                query_lower = query.lower()
                text_candidates = set()
                for path_str in candidates:
                    if query_lower in Path(path_str).name.lower():
                        text_candidates.add(path_str)
                candidates = text_candidates
            
            # Convert to metadata objects
            for path_str in candidates:
                results.append(self.index[path_str])
            
            return results
    
    def find_duplicates(self) -> List[List[FileMetadata]]:
        """Find duplicate files by checksum"""
        with self.lock:
            duplicates = []
            for checksum, paths in self.checksum_index.items():
                if len(paths) > 1:
                    duplicate_group = [self.index[path] for path in paths]
                    duplicates.append(duplicate_group)
            return duplicates
    
    def _get_size_range(self, size: int) -> Tuple[int, int]:
        """Get size range category"""
        if size < 1024:  # < 1KB
            return (0, 1023)
        elif size < 1024 * 1024:  # < 1MB
            return (1024, 1024 * 1024 - 1)
        elif size < 1024 * 1024 * 1024:  # < 1GB
            return (1024 * 1024, 1024 * 1024 * 1024 - 1)
        else:  # >= 1GB
            return (1024 * 1024 * 1024, float('inf'))
    
    def save_index(self):
        """Save index to disk"""
        try:
            with self.lock:
                # Convert to serializable format
                index_data = {}
                for path_str, metadata in self.index.items():
                    index_data[path_str] = {
                        'size': metadata.size,
                        'created_time': metadata.created_time,
                        'modified_time': metadata.modified_time,
                        'accessed_time': metadata.accessed_time,
                        'file_type': metadata.file_type.name,
                        'mime_type': metadata.mime_type,
                        'checksum': metadata.checksum,
                        'permissions': metadata.permissions,
                        'owner': metadata.owner,
                        'group': metadata.group,
                        'is_symlink': metadata.is_symlink,
                        'target_path': str(metadata.target_path) if metadata.target_path else None,
                        'encoding': metadata.encoding,
                        'line_count': metadata.line_count,
                        'version': metadata.version,
                        'tags': list(metadata.tags),
                        'custom_metadata': metadata.custom_metadata
                    }
                
                with open(self.index_path, 'w') as f:
                    json.dump(index_data, f, indent=2)
        except Exception as e:
            print(f"Error saving file index: {e}")
    
    def load_index(self):
        """Load index from disk"""
        try:
            if self.index_path.exists():
                with open(self.index_path, 'r') as f:
                    index_data = json.load(f)
                
                with self.lock:
                    for path_str, data in index_data.items():
                        metadata = FileMetadata(
                            path=Path(path_str),
                            size=data['size'],
                            created_time=data['created_time'],
                            modified_time=data['modified_time'],
                            accessed_time=data['accessed_time'],
                            file_type=FileType[data['file_type']],
                            mime_type=data['mime_type'],
                            checksum=data['checksum'],
                            permissions=data['permissions'],
                            owner=data['owner'],
                            group=data['group'],
                            is_symlink=data['is_symlink'],
                            target_path=Path(data['target_path']) if data['target_path'] else None,
                            encoding=data['encoding'],
                            line_count=data['line_count'],
                            version=data['version'],
                            tags=set(data['tags']),
                            custom_metadata=data['custom_metadata']
                        )
                        self.add_file(metadata)
        except Exception as e:
            print(f"Error loading file index: {e}")


class FileWatcher(FileSystemEventHandler):
    """File system event watcher"""
    
    def __init__(self, file_manager):
        super().__init__()
        self.file_manager = file_manager
        self.event_queue = asyncio.Queue()
    
    def on_any_event(self, event):
        """Handle any file system event"""
        try:
            asyncio.create_task(self.event_queue.put({
                'type': event.event_type,
                'path': event.src_path,
                'is_directory': event.is_directory,
                'timestamp': time.time()
            }))
        except:
            pass  # Queue might be full or event loop not running
    
    async def process_events(self):
        """Process file system events"""
        while True:
            try:
                event = await self.event_queue.get()
                await self.file_manager._handle_fs_event(event)
            except Exception as e:
                print(f"Error processing file system event: {e}")


class FileManager:
    """Advanced file management system"""
    
    def __init__(self, app):
        self.app = app
        self.base_path = Path.cwd()
        self.data_dir = self.base_path / "data"
        self.temp_dir = self.base_path / "temp"
        self.backup_dir = self.base_path / "backups"
        self.index_dir = self.base_path / "indexes"
        
        # File operations
        self.operation_history = deque(maxlen=1000)
        self.pending_operations = []
        self.batch_operations = {}
        
        # File monitoring
        self.file_index = None
        self.file_watcher = None
        self.observer = None
        self.watched_paths = set()
        
        # Versioning
        self.versioning_enabled = True
        self.max_versions = 10
        self.version_storage = {}  # file_path -> List[FileVersion]
        
        # Security and validation
        self.allowed_extensions = set()
        self.blocked_extensions = {'.exe', '.bat', '.cmd', '.scr', '.vbs'}
        self.max_file_size = 100 * 1024 * 1024  # 100MB
        self.virus_scan_enabled = False
        
        # Performance
        self.cache = {}
        self.cache_max_size = 1000
        self.enable_compression = True
        
        # Statistics
        self.stats = {
            'files_created': 0,
            'files_read': 0,
            'files_written': 0,
            'files_deleted': 0,
            'bytes_read': 0,
            'bytes_written': 0,
            'operations_count': 0,
            'errors_count': 0
        }
    
    async def setup(self) -> bool:
        """Setup file manager"""
        try:
            # Create directories
            directories = [
                self.data_dir,
                self.temp_dir,
                self.backup_dir,
                self.index_dir
            ]
            
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
            
            # Initialize file index
            index_path = self.index_dir / "file_index.json"
            self.file_index = FileIndex(index_path)
            
            # Setup file watcher
            self.file_watcher = FileWatcher(self)
            self.observer = Observer()
            
            # Start monitoring base directories
            for directory in directories:
                if directory.exists():
                    self.observer.schedule(self.file_watcher, str(directory), recursive=True)
                    self.watched_paths.add(directory)
            
            self.observer.start()
            
            # Load configuration
            await self._load_configuration()
            
            self.app.logger.info("File manager initialized successfully")
            return True
            
        except Exception as e:
            self.app.error_handler.handle_error(e, "FileManager.setup")
            return False
    
    async def _load_configuration(self):
        """Load file manager configuration"""
        try:
            config_file = self.app.config.get('file_manager.config_file', 'config/file_manager.json')
            config_path = Path(config_file)
            
            if config_path.exists():
                content = await self.read_text_async(config_path)
                if content:
                    config = json.loads(content)
                    
                    self.max_file_size = config.get('max_file_size', self.max_file_size)
                    self.max_versions = config.get('max_versions', self.max_versions)
                    self.versioning_enabled = config.get('versioning_enabled', self.versioning_enabled)
                    self.allowed_extensions = set(config.get('allowed_extensions', []))
                    self.blocked_extensions.update(config.get('blocked_extensions', []))
                    self.virus_scan_enabled = config.get('virus_scan_enabled', False)
        except Exception as e:
            self.app.logger.warning(f"Could not load file manager configuration: {e}")
    
    # Core file operations
    async def read_text_async(self, file_path: Union[str, Path], encoding: str = 'utf-8') -> Optional[str]:
        """Read text file asynchronously"""
        try:
            path = Path(file_path)
            
            # Security check
            if not await self._validate_file_access(path, FileOperation.READ):
                return None
            
            # Check cache
            cache_key = f"read_text:{path}:{path.stat().st_mtime}"
            if cache_key in self.cache:
                self.stats['files_read'] += 1
                return self.cache[cache_key]
            
            # Read file
            loop = asyncio.get_event_loop()
            content = await loop.run_in_executor(None, lambda: path.read_text(encoding=encoding))
            
            # Update cache
            if len(self.cache) < self.cache_max_size:
                self.cache[cache_key] = content
            
            # Update statistics
            self.stats['files_read'] += 1
            self.stats['bytes_read'] += len(content.encode())
            
            # Update file index
            if self.file_index:
                metadata = await self._generate_file_metadata(path)
                self.file_index.add_file(metadata)
            
            # Record operation
            await self._record_operation(FileOperation.READ, path, success=True)
            
            return content
            
        except Exception as e:
            await self._record_operation(FileOperation.READ, Path(file_path), success=False, error=str(e))
            self.app.error_handler.handle_error(e, f"FileManager.read_text_async({file_path})")
            return None
    
    async def write_text_async(self, file_path: Union[str, Path], content: str, 
                              encoding: str = 'utf-8', create_backup: bool = True) -> bool:
        """Write text file asynchronously"""
        try:
            path = Path(file_path)
            
            # Security check
            if not await self._validate_file_access(path, FileOperation.WRITE):
                return False
            
            # Create backup if file exists and versioning is enabled
            if create_backup and self.versioning_enabled and path.exists():
                await self._create_backup(path)
            
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write file
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: path.write_text(content, encoding=encoding))
            
            # Update statistics
            self.stats['files_written'] += 1
            self.stats['bytes_written'] += len(content.encode())
            
            if not path.existed_before_write:
                self.stats['files_created'] += 1
            
            # Update file index
            if self.file_index:
                metadata = await self._generate_file_metadata(path)
                self.file_index.add_file(metadata)
            
            # Clear cache
            self._clear_file_cache(path)
            
            # Record operation
            await self._record_operation(FileOperation.WRITE, path, success=True)
            
            return True
            
        except Exception as e:
            await self._record_operation(FileOperation.WRITE, Path(file_path), success=False, error=str(e))
            self.app.error_handler.handle_error(e, f"FileManager.write_text_async({file_path})")
            return False
    
    async def read_binary_async(self, file_path: Union[str, Path]) -> Optional[bytes]:
        """Read binary file asynchronously"""
        try:
            path = Path(file_path)
            
            # Security check
            if not await self._validate_file_access(path, FileOperation.READ):
                return None
            
            # Read file
            loop = asyncio.get_event_loop()
            content = await loop.run_in_executor(None, lambda: path.read_bytes())
            
            # Update statistics
            self.stats['files_read'] += 1
            self.stats['bytes_read'] += len(content)
            
            # Record operation
            await self._record_operation(FileOperation.READ, path, success=True)
            
            return content
            
        except Exception as e:
            await self._record_operation(FileOperation.READ, Path(file_path), success=False, error=str(e))
            self.app.error_handler.handle_error(e, f"FileManager.read_binary_async({file_path})")
            return None
    
    async def write_binary_async(self, file_path: Union[str, Path], content: bytes,
                                create_backup: bool = True) -> bool:
        """Write binary file asynchronously"""
        try:
            path = Path(file_path)
            
            # Security check
            if not await self._validate_file_access(path, FileOperation.WRITE):
                return False
            
            # Create backup if file exists and versioning is enabled
            if create_backup and self.versioning_enabled and path.exists():
                await self._create_backup(path)
            
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write file
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: path.write_bytes(content))
            
            # Update statistics
            self.stats['files_written'] += 1
            self.stats['bytes_written'] += len(content)
            
            # Update file index
            if self.file_index:
                metadata = await self._generate_file_metadata(path)
                self.file_index.add_file(metadata)
            
            # Record operation
            await self._record_operation(FileOperation.WRITE, path, success=True)
            
            return True
            
        except Exception as e:
            await self._record_operation(FileOperation.WRITE, Path(file_path), success=False, error=str(e))
            self.app.error_handler.handle_error(e, f"FileManager.write_binary_async({file_path})")
            return False
    
    async def delete_async(self, file_path: Union[str, Path], secure_delete: bool = False) -> bool:
        """Delete file asynchronously"""
        try:
            path = Path(file_path)
            
            # Security check
            if not await self._validate_file_access(path, FileOperation.DELETE):
                return False
            
            if not path.exists():
                return True
            
            # Create backup before deletion if versioning is enabled
            if self.versioning_enabled:
                await self._create_backup(path)
            
            # Secure deletion (overwrite with random data)
            if secure_delete and path.is_file():
                await self._secure_delete(path)
            else:
                # Regular deletion
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path)
            
            # Update statistics
            self.stats['files_deleted'] += 1
            
            # Remove from index
            if self.file_index:
                self.file_index.remove_file(path)
            
            # Clear cache
            self._clear_file_cache(path)
            
            # Record operation
            await self._record_operation(FileOperation.DELETE, path, success=True)
            
            return True
            
        except Exception as e:
            await self._record_operation(FileOperation.DELETE, Path(file_path), success=False, error=str(e))
            self.app.error_handler.handle_error(e, f"FileManager.delete_async({file_path})")
            return False
    
    async def copy_async(self, source: Union[str, Path], destination: Union[str, Path],
                        preserve_metadata: bool = True) -> bool:
        """Copy file or directory asynchronously"""
        try:
            src_path = Path(source)
            dst_path = Path(destination)
            
            # Security checks
            if not await self._validate_file_access(src_path, FileOperation.READ):
                return False
            if not await self._validate_file_access(dst_path, FileOperation.WRITE):
                return False
            
            # Ensure destination directory exists
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy operation
            loop = asyncio.get_event_loop()
            if src_path.is_file():
                await loop.run_in_executor(None, lambda: shutil.copy2(src_path, dst_path) if preserve_metadata else shutil.copy(src_path, dst_path))
            elif src_path.is_dir():
                await loop.run_in_executor(None, lambda: shutil.copytree(src_path, dst_path, dirs_exist_ok=True))
            
            # Update file index
            if self.file_index and dst_path.is_file():
                metadata = await self._generate_file_metadata(dst_path)
                self.file_index.add_file(metadata)
            
            # Record operation
            await self._record_operation(FileOperation.COPY, src_path, success=True, extra_info={'destination': str(dst_path)})
            
            return True
            
        except Exception as e:
            await self._record_operation(FileOperation.COPY, Path(source), success=False, error=str(e))
            self.app.error_handler.handle_error(e, f"FileManager.copy_async({source} -> {destination})")
            return False
    
    async def move_async(self, source: Union[str, Path], destination: Union[str, Path]) -> bool:
        """Move file or directory asynchronously"""
        try:
            src_path = Path(source)
            dst_path = Path(destination)
            
            # Security checks
            if not await self._validate_file_access(src_path, FileOperation.READ):
                return False
            if not await self._validate_file_access(dst_path, FileOperation.WRITE):
                return False
            
            # Ensure destination directory exists
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Move operation
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: shutil.move(str(src_path), str(dst_path)))
            
            # Update file index
            if self.file_index:
                self.file_index.remove_file(src_path)
                if dst_path.is_file():
                    metadata = await self._generate_file_metadata(dst_path)
                    self.file_index.add_file(metadata)
            
            # Clear cache
            self._clear_file_cache(src_path)
            
            # Record operation
            await self._record_operation(FileOperation.MOVE, src_path, success=True, extra_info={'destination': str(dst_path)})
            
            return True
            
        except Exception as e:
            await self._record_operation(FileOperation.MOVE, Path(source), success=False, error=str(e))
            self.app.error_handler.handle_error(e, f"FileManager.move_async({source} -> {destination})")
            return False
    
    # Compression and archiving
    async def compress_file_async(self, file_path: Union[str, Path], 
                                 compression_type: str = 'gzip') -> Optional[Path]:
        """Compress file asynchronously"""
        try:
            path = Path(file_path)
            
            if not path.exists() or not path.is_file():
                return None
            
            # Determine output path
            if compression_type == 'gzip':
                output_path = path.with_suffix(path.suffix + '.gz')
                
                # Compress
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._gzip_compress, path, output_path)
            else:
                raise ValueError(f"Unsupported compression type: {compression_type}")
            
            # Record operation
            await self._record_operation(FileOperation.COMPRESS, path, success=True, 
                                       extra_info={'output': str(output_path), 'type': compression_type})
            
            return output_path
            
        except Exception as e:
            await self._record_operation(FileOperation.COMPRESS, Path(file_path), success=False, error=str(e))
            self.app.error_handler.handle_error(e, f"FileManager.compress_file_async({file_path})")
            return None
    
    def _gzip_compress(self, input_path: Path, output_path: Path):
        """Compress file using gzip"""
        with open(input_path, 'rb') as f_in:
            with gzip.open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    
    # File validation and security
    async def _validate_file_access(self, path: Path, operation: FileOperation) -> bool:
        """Validate file access for security"""
        try:
            # Check if path is within allowed directories
            resolved_path = path.resolve()
            base_resolved = self.base_path.resolve()
            
            if not str(resolved_path).startswith(str(base_resolved)):
                self.app.logger.warning(f"Access denied: {path} is outside base directory")
                return False
            
            # Check file extension for write operations
            if operation in [FileOperation.WRITE, FileOperation.CREATE]:
                if path.suffix.lower() in self.blocked_extensions:
                    self.app.logger.warning(f"Access denied: {path.suffix} extension is blocked")
                    return False
                
                if self.allowed_extensions and path.suffix.lower() not in self.allowed_extensions:
                    self.app.logger.warning(f"Access denied: {path.suffix} extension is not allowed")
                    return False
            
            # Check file size for write operations
            if operation == FileOperation.WRITE and path.exists():
                if path.stat().st_size > self.max_file_size:
                    self.app.logger.warning(f"Access denied: {path} exceeds maximum file size")
                    return False
            
            return True
            
        except Exception as e:
            self.app.error_handler.handle_error(e, f"FileManager._validate_file_access({path}, {operation})")
            return False
    
    async def _generate_file_metadata(self, path: Path) -> FileMetadata:
        """Generate comprehensive file metadata"""
        try:
            stat_info = path.stat()
            
            metadata = FileMetadata(
                path=path,
                size=stat_info.st_size,
                created_time=stat_info.st_ctime,
                modified_time=stat_info.st_mtime,
                accessed_time=stat_info.st_atime,
                permissions=stat.filemode(stat_info.st_mode),
                is_symlink=path.is_symlink()
            )
            
            if path.is_symlink():
                metadata.target_path = path.readlink()
            
            # Determine file type
            metadata.file_type = self._classify_file_type(path)
            
            # Get MIME type
            metadata.mime_type = mimetypes.guess_type(str(path))[0] or 'application/octet-stream'
            
            # Calculate checksum for regular files
            if path.is_file() and stat_info.st_size < 10 * 1024 * 1024:  # Only for files < 10MB
                loop = asyncio.get_event_loop()
                metadata.checksum = await loop.run_in_executor(None, self._calculate_checksum, path)
            
            # Get encoding for text files
            if metadata.file_type == FileType.TEXT:
                metadata.encoding = await self._detect_encoding(path)
                
                # Count lines for small text files
                if stat_info.st_size < 1024 * 1024:  # < 1MB
                    metadata.line_count = await self._count_lines(path)
            
            # Get owner information (Unix systems)
            try:
                import pwd
                metadata.owner = pwd.getpwuid(stat_info.st_uid).pw_name
            except (ImportError, KeyError):
                metadata.owner = str(stat_info.st_uid)
            
            try:
                import grp
                metadata.group = grp.getgrgid(stat_info.st_gid).gr_name
            except (ImportError, KeyError):
                metadata.group = str(stat_info.st_gid)
            
            return metadata
            
        except Exception as e:
            self.app.error_handler.handle_error(e, f"FileManager._generate_file_metadata({path})")
            # Return basic metadata on error
            return FileMetadata(path=path)
    
    def _classify_file_type(self, path: Path) -> FileType:
        """Classify file type based on extension and content"""
        suffix = path.suffix.lower()
        
        # Text files
        text_extensions = {'.txt', '.md', '.rst', '.log', '.cfg', '.ini', '.json', '.xml', '.yaml', '.yml', '.csv', '.tsv'}
        if suffix in text_extensions:
            return FileType.TEXT
        
        # Configuration files
        config_extensions = {'.conf', '.config', '.ini', '.cfg', '.properties', '.toml'}
        if suffix in config_extensions:
            return FileType.CONFIG
        
        # Images
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.svg', '.webp'}
        if suffix in image_extensions:
            return FileType.IMAGE
        
        # Videos
        video_extensions = {'.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mkv'}
        if suffix in video_extensions:
            return FileType.VIDEO
        
        # Audio
        audio_extensions = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a'}
        if suffix in audio_extensions:
            return FileType.AUDIO
        
        # Documents
        doc_extensions = {'.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.odt', '.ods', '.odp'}
        if suffix in doc_extensions:
            return FileType.DOCUMENT
        
        # Archives
        archive_extensions = {'.zip', '.tar', '.gz', '.bz2', '.xz', '.7z', '.rar'}
        if suffix in archive_extensions:
            return FileType.ARCHIVE
        
        # Executables
        exec_extensions = {'.exe', '.bat', '.cmd', '.sh', '.bin', '.app', '.deb', '.rpm'}
        if suffix in exec_extensions:
            return FileType.EXECUTABLE
        
        # Databases
        db_extensions = {'.db', '.sqlite', '.sqlite3', '.mdb', '.accdb'}
        if suffix in db_extensions:
            return FileType.DATABASE
        
        # Temporary files
        temp_extensions = {'.tmp', '.temp', '.bak', '.swp', '.swo'}
        if suffix in temp_extensions or path.name.startswith('.'):
            return FileType.TEMPORARY
        
        # Default to binary for unknown types
        return FileType.BINARY
    
    def _calculate_checksum(self, path: Path, algorithm: str = 'sha256') -> str:
        """Calculate file checksum"""
        hash_obj = hashlib.new(algorithm)
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()
    
    async def _detect_encoding(self, path: Path) -> Optional[str]:
        """Detect text file encoding"""
        try:
            # Try common encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'ascii']
            
            for encoding in encodings:
                try:
                    with open(path, 'r', encoding=encoding) as f:
                        f.read(1024)  # Read first 1KB
                    return encoding
                except UnicodeDecodeError:
                    continue
            
            return None
        except Exception:
            return None
    
    async def _count_lines(self, path: Path) -> Optional[int]:
        """Count lines in text file"""
        try:
            loop = asyncio.get_event_loop()
            count = await loop.run_in_executor(None, self._count_lines_sync, path)
            return count
        except Exception:
            return None
    
    def _count_lines_sync(self, path: Path) -> int:
        """Count lines synchronously"""
        with open(path, 'rb') as f:
            count = sum(1 for _ in f)
        return count
    
    # Backup and versioning
    async def _create_backup(self, path: Path):
        """Create backup of file"""
        try:
            if not path.exists():
                return
            
            # Generate backup path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{path.name}.{timestamp}.backup"
            backup_path = self.backup_dir / path.parent.relative_to(self.base_path) / backup_name
            
            # Ensure backup directory exists
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file to backup location
            shutil.copy2(path, backup_path)
            
            # Update version storage
            file_str = str(path)
            if file_str not in self.version_storage:
                self.version_storage[file_str] = []
            
            version = FileVersion(
                version=len(self.version_storage[file_str]) + 1,
                timestamp=time.time(),
                size=path.stat().st_size,
                checksum=self._calculate_checksum(path),
                backup_path=backup_path
            )
            
            self.version_storage[file_str].append(version)
            
            # Cleanup old versions
            if len(self.version_storage[file_str]) > self.max_versions:
                old_version = self.version_storage[file_str].pop(0)
                if old_version.backup_path and old_version.backup_path.exists():
                    old_version.backup_path.unlink()
            
        except Exception as e:
            self.app.error_handler.handle_error(e, f"FileManager._create_backup({path})")
    
    async def _secure_delete(self, path: Path):
        """Securely delete file by overwriting with random data"""
        try:
            if not path.is_file():
                return
            
            file_size = path.stat().st_size
            
            # Overwrite with random data multiple times
            for _ in range(3):
                with open(path, 'wb') as f:
                    while f.tell() < file_size:
                        chunk_size = min(8192, file_size - f.tell())
                        random_data = os.urandom(chunk_size)
                        f.write(random_data)
                f.flush()
                os.fsync(f.fileno())
            
            # Finally delete the file
            path.unlink()
            
        except Exception as e:
            self.app.error_handler.handle_error(e, f"FileManager._secure_delete({path})")
    
    # Event handling
    async def _handle_fs_event(self, event: Dict[str, Any]):
        """Handle file system event"""
        try:
            path = Path(event['path'])
            event_type = event['type']
            
            if event['is_directory']:
                return  # Skip directory events for now
            
            # Update file index based on event type
            if self.file_index:
                if event_type in ['created', 'modified']:
                    if path.exists():
                        metadata = await self._generate_file_metadata(path)
                        self.file_index.add_file(metadata)
                elif event_type == 'deleted':
                    self.file_index.remove_file(path)
            
            # Clear cache for affected file
            self._clear_file_cache(path)
            
        except Exception as e:
            self.app.error_handler.handle_error(e, f"FileManager._handle_fs_event({event})")
    
    # Utility methods
    def _clear_file_cache(self, path: Path):
        """Clear cache entries for a file"""
        path_str = str(path)
        keys_to_remove = [key for key in self.cache.keys() if path_str in key]
        for key in keys_to_remove:
            del self.cache[key]
    
    async def _record_operation(self, operation: FileOperation, path: Path, 
                               success: bool, error: str = None, extra_info: Dict = None):
        """Record file operation for auditing"""
        record = {
            'timestamp': time.time(),
            'operation': operation.name,
            'path': str(path),
            'success': success,
            'error': error,
            'extra_info': extra_info or {}
        }
        
        self.operation_history.append(record)
        self.stats['operations_count'] += 1
        
        if not success:
            self.stats['errors_count'] += 1
    
    # Public API methods
    def get_file_info(self, file_path: Union[str, Path]) -> Optional[FileMetadata]:
        """Get file information from index"""
        if self.file_index:
            path_str = str(Path(file_path))
            return self.file_index.index.get(path_str)
        return None
    
    def search_files(self, **criteria) -> List[FileMetadata]:
        """Search files using various criteria"""
        if self.file_index:
            return self.file_index.search_files(**criteria)
        return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get file manager statistics"""
        stats = self.stats.copy()
        
        if self.file_index:
            stats['indexed_files'] = len(self.file_index.index)
            stats['unique_checksums'] = len(self.file_index.checksum_index)
        
        stats['watched_paths'] = len(self.watched_paths)
        stats['cached_files'] = len(self.cache)
        stats['operation_history_size'] = len(self.operation_history)
        
        return stats
    
    async def cleanup(self):
        """Cleanup file manager resources"""
        try:
            # Stop file watcher
            if self.observer:
                self.observer.stop()
                self.observer.join()
            
            # Save file index
            if self.file_index:
                self.file_index.save_index()
            
            # Clear cache
            self.cache.clear()
            
            self.app.logger.info("File manager cleanup completed")
            
        except Exception as e:
            self.app.error_handler.handle_error(e, "FileManager.cleanup")


class DirectoryManager:
    """Advanced directory management system"""
    
    def __init__(self, app):
        self.app = app
        self.base_path = Path.cwd()
        self.monitored_directories = {}
        self.directory_stats = {}
        self.sync_tasks = {}
        
    async def setup(self) -> bool:
        """Setup directory manager"""
        try:
            # Initialize monitoring for key directories
            key_dirs = ['data', 'logs', 'config', 'temp', 'backups']
            
            for dir_name in key_dirs:
                dir_path = self.base_path / dir_name
                if dir_path.exists():
                    await self.monitor_directory(dir_path)
            
            self.app.logger.info("Directory manager initialized successfully")
            return True
            
        except Exception as e:
            self.app.error_handler.handle_error(e, "DirectoryManager.setup")
            return False
    
    async def monitor_directory(self, path: Path, recursive: bool = True):
        """Start monitoring a directory"""
        try:
            path_str = str(path)
            
            if path_str in self.monitored_directories:
                return  # Already monitoring
            
            # Create monitoring entry
            self.monitored_directories[path_str] = {
                'path': path,
                'recursive': recursive,
                'start_time': time.time(),
                'file_count': 0,
                'total_size': 0,
                'last_scan': None
            }
            
            # Perform initial scan
            await self._scan_directory(path, recursive)
            
        except Exception as e:
            self.app.error_handler.handle_error(e, f"DirectoryManager.monitor_directory({path})")
    
    async def _scan_directory(self, path: Path, recursive: bool = True):
        """Scan directory and update statistics"""
        try:
            file_count = 0
            total_size = 0
            
            if recursive:
                for item in path.rglob('*'):
                    if item.is_file():
                        file_count += 1
                        total_size += item.stat().st_size
            else:
                for item in path.iterdir():
                    if item.is_file():
                        file_count += 1
                        total_size += item.stat().st_size
            
            # Update monitoring data
            path_str = str(path)
            if path_str in self.monitored_directories:
                self.monitored_directories[path_str].update({
                    'file_count': file_count,
                    'total_size': total_size,
                    'last_scan': time.time()
                })
            
        except Exception as e:
            self.app.error_handler.handle_error(e, f"DirectoryManager._scan_directory({path})")
    
    def get_directory_stats(self, path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Get directory statistics"""
        path_str = str(Path(path))
        return self.monitored_directories.get(path_str)
    
    def get_all_monitored_directories(self) -> Dict[str, Dict[str, Any]]:
        """Get all monitored directories"""
        return self.monitored_directories.copy()
    
    async def cleanup(self):
        """Cleanup directory manager resources"""
        try:
            # Stop any running sync tasks
            for task in self.sync_tasks.values():
                task.cancel()
            
            if self.sync_tasks:
                await asyncio.gather(*self.sync_tasks.values(), return_exceptions=True)
            
            self.app.logger.info("Directory manager cleanup completed")
            
        except Exception as e:
            self.app.error_handler.handle_error(e, "DirectoryManager.cleanup")