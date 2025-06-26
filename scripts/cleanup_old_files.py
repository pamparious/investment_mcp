#!/usr/bin/env python3
"""
Cleanup script for Investment MCP System.

This script removes redundant files and directories identified during
the codebase cleanup and consolidation process.
"""

import os
import shutil
from pathlib import Path
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InvestmentMCPCleaner:
    """Cleaner for redundant files and directories."""
    
    def __init__(self, project_root: Path, dry_run: bool = True):
        self.project_root = Path(project_root)
        self.dry_run = dry_run
        self.removed_files = []
        self.removed_dirs = []
        self.preserved_files = []
    
    def clean_redundant_files(self):
        """Remove redundant files identified during analysis."""
        
        logger.info("Starting cleanup of redundant files...")
        
        # Files to remove (empty or redundant)
        files_to_remove = [
            # Empty __init__.py files that were creating import issues
            "backend/models/__init__.py",
            "backend/analysis/__init__.py", 
            "backend/mcp_agents/__init__.py",
            "mcp_servers/investment_server/__init__.py",
            
            # Empty analysis file
            "backend/analysis/valuation.py",
            
            # Empty MCP agent
            "backend/mcp_agents/portfolio_agent.py",
            
            # Redundant configuration files (keeping the unified one)
            # Note: We'll move useful content to unified config before deleting
        ]
        
        # Directories to remove (if empty after file cleanup)
        dirs_to_remove_if_empty = [
            "backend/models",  # Empty directory
            "src/agents",      # Experimental directory
            "src/api/v1/endpoints",  # Incomplete API
            "src/api/v1/middleware",
            "src/api/v1/schemas",
            "src/services",    # Empty services
        ]
        
        # Remove files
        for file_path in files_to_remove:
            self._remove_file(file_path)
        
        # Remove empty directories
        for dir_path in dirs_to_remove_if_empty:
            self._remove_directory_if_empty(dir_path)
        
        # Clean up specific redundant implementations
        self._cleanup_redundant_implementations()
        
        logger.info("Cleanup completed")
    
    def _remove_file(self, file_path: str):
        """Remove a single file."""
        
        full_path = self.project_root / file_path
        
        if full_path.exists():
            if full_path.is_file():
                if self.dry_run:
                    logger.info(f"[DRY RUN] Would remove file: {file_path}")
                else:
                    full_path.unlink()
                    logger.info(f"Removed file: {file_path}")
                    self.removed_files.append(str(file_path))
            else:
                logger.warning(f"Path is not a file: {file_path}")
        else:
            logger.debug(f"File not found (already removed?): {file_path}")
    
    def _remove_directory_if_empty(self, dir_path: str):
        """Remove directory if it's empty."""
        
        full_path = self.project_root / dir_path
        
        if full_path.exists() and full_path.is_dir():
            try:
                # Check if directory is empty
                if not any(full_path.iterdir()):
                    if self.dry_run:
                        logger.info(f"[DRY RUN] Would remove empty directory: {dir_path}")
                    else:
                        full_path.rmdir()
                        logger.info(f"Removed empty directory: {dir_path}")
                        self.removed_dirs.append(str(dir_path))
                else:
                    logger.debug(f"Directory not empty, preserving: {dir_path}")
            except OSError as e:
                logger.error(f"Error removing directory {dir_path}: {e}")
    
    def _cleanup_redundant_implementations(self):
        """Clean up redundant implementations and move to archive."""
        
        logger.info("Cleaning up redundant implementations...")
        
        # Create archive directory for old implementations
        archive_dir = self.project_root / "archive"
        
        if not self.dry_run:
            archive_dir.mkdir(exist_ok=True)
        
        # Move experimental src/ directory to archive (keep for reference)
        experimental_src = self.project_root / "src" / "agents"
        if experimental_src.exists():
            if self.dry_run:
                logger.info("[DRY RUN] Would archive experimental src/agents/")
            else:
                try:
                    # Move to archive if not dry run
                    archive_target = archive_dir / "experimental_agents"
                    if archive_target.exists():
                        shutil.rmtree(archive_target)
                    shutil.move(str(experimental_src), str(archive_target))
                    logger.info("Archived experimental agents to archive/")
                except Exception as e:
                    logger.error(f"Error archiving experimental code: {e}")
        
        # Archive incomplete API implementation
        incomplete_api = self.project_root / "src" / "api"
        if incomplete_api.exists():
            if self.dry_run:
                logger.info("[DRY RUN] Would archive incomplete API implementation")
            else:
                try:
                    archive_target = archive_dir / "incomplete_api"
                    if archive_target.exists():
                        shutil.rmtree(archive_target)
                    shutil.move(str(incomplete_api), str(archive_target))
                    logger.info("Archived incomplete API to archive/")
                except Exception as e:
                    logger.error(f"Error archiving incomplete API: {e}")
    
    def consolidate_requirements(self):
        """Consolidate requirements files."""
        
        logger.info("Consolidating requirements files...")
        
        # Main requirements files
        main_req = self.project_root / "requirements.txt"
        api_req = self.project_root / "investment_mcp_api" / "requirements.txt"
        unified_req = self.project_root / "requirements_unified.txt"
        
        if not self.dry_run and unified_req.exists():
            # Backup old requirements
            if main_req.exists():
                backup_path = self.project_root / "requirements_old.txt"
                shutil.copy2(main_req, backup_path)
                logger.info("Backed up old requirements.txt")
            
            # Replace main requirements with unified version
            shutil.copy2(unified_req, main_req)
            logger.info("Updated main requirements.txt with unified version")
        else:
            logger.info("[DRY RUN] Would consolidate requirements files")
    
    def verify_essential_files(self):
        """Verify that essential files are preserved."""
        
        logger.info("Verifying essential files are preserved...")
        
        essential_files = [
            "src/core/config.py",
            "src/core/database.py", 
            "src/collectors/market_data.py",
            "src/collectors/swedish_data.py",
            "src/analysis/risk.py",
            "src/analysis/technical.py",
            "src/analysis/portfolio.py",
            "src/ai/providers.py",
            "src/ai/portfolio_advisor.py",
            "src/mcp/server.py",
            "src/mcp/tools.py",
            "src/utils/logging.py",
            "src/utils/helpers.py",
            "src/__init__.py"
        ]
        
        missing_files = []
        
        for file_path in essential_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                missing_files.append(file_path)
            else:
                self.preserved_files.append(file_path)
        
        if missing_files:
            logger.error(f"CRITICAL: Missing essential files: {missing_files}")
            return False
        else:
            logger.info(f"✓ All {len(essential_files)} essential files preserved")
            return True
    
    def generate_cleanup_report(self):
        """Generate cleanup report."""
        
        logger.info("\n" + "="*60)
        logger.info("CLEANUP REPORT")
        logger.info("="*60)
        
        logger.info(f"Mode: {'DRY RUN' if self.dry_run else 'ACTUAL CLEANUP'}")
        logger.info(f"Files removed: {len(self.removed_files)}")
        logger.info(f"Directories removed: {len(self.removed_dirs)}")
        logger.info(f"Essential files preserved: {len(self.preserved_files)}")
        
        if self.removed_files:
            logger.info("\nRemoved files:")
            for file_path in self.removed_files:
                logger.info(f"  - {file_path}")
        
        if self.removed_dirs:
            logger.info("\nRemoved directories:")
            for dir_path in self.removed_dirs:
                logger.info(f"  - {dir_path}")
        
        logger.info("\n" + "="*60)
    
    def estimate_space_savings(self):
        """Estimate space savings from cleanup."""
        
        total_size = 0
        file_count = 0
        
        # Calculate size of files to be removed
        files_to_check = [
            "backend/models/__init__.py",
            "backend/analysis/__init__.py",
            "backend/mcp_agents/__init__.py",
            "backend/analysis/valuation.py",
            "backend/mcp_agents/portfolio_agent.py"
        ]
        
        for file_path in files_to_check:
            full_path = self.project_root / file_path
            if full_path.exists() and full_path.is_file():
                size = full_path.stat().st_size
                total_size += size
                file_count += 1
        
        # Add directory sizes (estimated)
        redundant_dirs = ["backend/models", "src/agents", "src/api"]
        for dir_path in redundant_dirs:
            full_path = self.project_root / dir_path
            if full_path.exists() and full_path.is_dir():
                for item in full_path.rglob("*"):
                    if item.is_file():
                        total_size += item.stat().st_size
                        file_count += 1
        
        logger.info(f"\nEstimated cleanup savings:")
        logger.info(f"  Files affected: {file_count}")
        logger.info(f"  Space saved: {total_size / 1024:.1f} KB")
        
        return total_size, file_count


def main():
    """Main cleanup execution."""
    
    parser = argparse.ArgumentParser(description="Clean up Investment MCP codebase")
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        default=True,
        help="Perform dry run (default: True)"
    )
    parser.add_argument(
        "--execute",
        action="store_true", 
        help="Actually perform cleanup (overrides --dry-run)"
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Project root directory (default: current directory)"
    )
    
    args = parser.parse_args()
    
    # Determine execution mode
    dry_run = not args.execute
    
    if not dry_run:
        response = input("\n⚠️  This will permanently delete files. Are you sure? (yes/no): ")
        if response.lower() != "yes":
            logger.info("Cleanup cancelled by user")
            return
    
    project_root = Path(args.project_root).resolve()
    
    if not project_root.exists():
        logger.error(f"Project root does not exist: {project_root}")
        return
    
    logger.info(f"Investment MCP Cleanup Tool")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Mode: {'DRY RUN' if dry_run else 'EXECUTE'}")
    
    # Create cleaner and run cleanup
    cleaner = InvestmentMCPCleaner(project_root, dry_run=dry_run)
    
    # Estimate space savings
    cleaner.estimate_space_savings()
    
    # Verify essential files exist before cleanup
    if not cleaner.verify_essential_files():
        logger.error("Essential files missing - aborting cleanup")
        return
    
    # Perform cleanup
    cleaner.clean_redundant_files()
    cleaner.consolidate_requirements()
    
    # Generate report
    cleaner.generate_cleanup_report()
    
    if dry_run:
        logger.info("\n💡 This was a dry run. Use --execute to perform actual cleanup.")
    else:
        logger.info("\n✅ Cleanup completed successfully!")


if __name__ == "__main__":
    main()