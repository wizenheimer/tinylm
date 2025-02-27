/**
 * FileProgressTracker - Manages per-file progress tracking
 * Tracks the status and progress of individual files during model loading
 */

import { FileInfo, OverallProgress } from './types';

/**
 * FileProgressTracker - Manages per-file progress tracking
 * Tracks the status and progress of individual files during model loading
 */
export class FileProgressTracker {
  private files: Map<string, FileInfo> = new Map();
  private totalBytes: number = 0;
  private loadedBytes: number = 0;

  /**
   * Register a new file for tracking
   * @param {string} fileId - Unique file identifier
   * @param {Partial<FileInfo>} info - File information
   * @returns {FileInfo} File information object
   */
  registerFile(fileId: string, info: Partial<FileInfo> = {}): FileInfo {
    if (!this.files.has(fileId)) {
      const fileInfo: FileInfo = {
        id: fileId,
        name: info.name || fileId,
        status: 'pending',
        progress: 0,
        percentComplete: 0,
        bytesLoaded: 0,
        bytesTotal: info.bytesTotal || 0,
        startTime: Date.now(),
        lastUpdateTime: Date.now(),
        speed: 0, // bytes per second
        timeRemaining: null, // in seconds
        ...info
      };

      this.files.set(fileId, fileInfo);

      if (fileInfo.bytesTotal > 0) {
        this.totalBytes += fileInfo.bytesTotal;
      }

      return fileInfo;
    }

    return this.files.get(fileId)!;
  }

  /**
   * Update progress for a specific file
   * @param {string} fileId - File identifier
   * @param {Partial<FileInfo>} update - Progress update
   * @returns {FileInfo} Updated file info
   */
  updateFile(fileId: string, update: Partial<FileInfo>): FileInfo {
    if (!this.files.has(fileId)) {
      return this.registerFile(fileId, update);
    }

    const fileInfo = this.files.get(fileId)!;
    const now = Date.now();
    const timeDelta = (now - fileInfo.lastUpdateTime) / 1000; // seconds

    // Calculate bytes delta for speed estimation
    let bytesDelta = 0;
    if (update.progress !== undefined && update.total !== undefined) {
      // If we have new progress/total info (compatibility with JS version)
      const newBytesLoaded = update.progress;
      bytesDelta = newBytesLoaded - fileInfo.bytesLoaded;

      // Update total bytes if it changed
      if (update.total !== fileInfo.bytesTotal && update.total > 0) {
        this.totalBytes = this.totalBytes - fileInfo.bytesTotal + update.total;
        fileInfo.bytesTotal = update.total;
      }

      // Update loaded bytes
      this.loadedBytes = this.loadedBytes - fileInfo.bytesLoaded + newBytesLoaded;
      fileInfo.bytesLoaded = newBytesLoaded;

      // Calculate percentage
      if (fileInfo.bytesTotal > 0) {
        fileInfo.progress = fileInfo.bytesLoaded / fileInfo.bytesTotal;
        fileInfo.percentComplete = Math.round(fileInfo.progress * 100);
      }
    } else if (update.bytesLoaded !== undefined && update.bytesTotal !== undefined) {
      // Support for explicit bytesLoaded/bytesTotal (TS version preference)
      const newBytesLoaded = update.bytesLoaded;
      bytesDelta = newBytesLoaded - fileInfo.bytesLoaded;

      // Update total bytes if it changed
      if (update.bytesTotal !== fileInfo.bytesTotal && update.bytesTotal > 0) {
        this.totalBytes = this.totalBytes - fileInfo.bytesTotal + update.bytesTotal;
        fileInfo.bytesTotal = update.bytesTotal;
      }

      // Update loaded bytes
      this.loadedBytes = this.loadedBytes - fileInfo.bytesLoaded + newBytesLoaded;
      fileInfo.bytesLoaded = newBytesLoaded;

      // Calculate percentage
      if (fileInfo.bytesTotal > 0) {
        fileInfo.progress = fileInfo.bytesLoaded / fileInfo.bytesTotal;
        fileInfo.percentComplete = Math.round(fileInfo.progress * 100);
      }
    }

    // Calculate speed (bytes per second) with some smoothing
    if (timeDelta > 0 && bytesDelta > 0) {
      const instantSpeed = bytesDelta / timeDelta;
      // Smooth speed calculation (70% previous, 30% new)
      fileInfo.speed = fileInfo.speed === 0
        ? instantSpeed
        : (fileInfo.speed * 0.7) + (instantSpeed * 0.3);

      // Calculate time remaining
      if (fileInfo.speed > 0 && fileInfo.bytesTotal > fileInfo.bytesLoaded) {
        const bytesRemaining = fileInfo.bytesTotal - fileInfo.bytesLoaded;
        fileInfo.timeRemaining = bytesRemaining / fileInfo.speed;
      }
    }

    // Update status
    if (update.status) {
      fileInfo.status = update.status;

      if (update.status === 'done' || update.status === 'complete') {
        fileInfo.progress = 1;
        fileInfo.percentComplete = 100;
        fileInfo.timeRemaining = 0;

        // Ensure consistency with bytes
        if (fileInfo.bytesTotal > 0 && fileInfo.bytesLoaded !== fileInfo.bytesTotal) {
          this.loadedBytes = this.loadedBytes - fileInfo.bytesLoaded + fileInfo.bytesTotal;
          fileInfo.bytesLoaded = fileInfo.bytesTotal;
        }
      }
    }

    // Update any other properties
    Object.assign(fileInfo, update);
    fileInfo.lastUpdateTime = now;

    return fileInfo;
  }

  /**
   * Mark a file as complete
   * @param {string} fileId - File identifier
   * @returns {FileInfo | null} Updated file info or null if not found
   */
  completeFile(fileId: string): FileInfo | null {
    if (!this.files.has(fileId)) {
      return null;
    }

    return this.updateFile(fileId, {
      status: 'done',
      progress: 1,
      percentComplete: 100,
      timeRemaining: 0
    });
  }

  /**
   * Get information for a specific file
   * @param {string} fileId - File identifier
   * @returns {FileInfo|null} File info or null if not found
   */
  getFile(fileId: string): FileInfo | null {
    return this.files.has(fileId) ? this.files.get(fileId)! : null;
  }

  /**
   * Get information about all tracked files
   * @returns {FileInfo[]} Array of file information objects
   */
  getAllFiles(): FileInfo[] {
    return Array.from(this.files.values());
  }

  /**
   * Get active (incomplete) files
   * @returns {FileInfo[]} Array of active file information objects
   */
  getActiveFiles(): FileInfo[] {
    return this.getAllFiles().filter(file =>
      file.status !== 'done' && file.status !== 'error');
  }

  /**
   * Get overall progress across all files
   * @returns {OverallProgress} Overall progress information
   */
  getOverallProgress(): OverallProgress {
    const files = this.getAllFiles();
    const activeFiles = this.getActiveFiles();

    // Calculate weighted overall progress
    let overallProgress = 0;
    if (this.totalBytes > 0) {
      overallProgress = this.loadedBytes / this.totalBytes;
    } else if (files.length > 0) {
      // Fallback to simple average if we don't have byte information
      const progressSum = files.reduce((sum, file) => sum + file.progress, 0);
      overallProgress = progressSum / files.length;
    }

    const percentComplete = Math.round(overallProgress * 100);

    // Calculate overall speed and time remaining
    let overallSpeed = 0;
    let maxTimeRemaining = 0;

    activeFiles.forEach(file => {
      overallSpeed += file.speed || 0;
      if (file.timeRemaining !== null && file.timeRemaining > maxTimeRemaining) {
        maxTimeRemaining = file.timeRemaining;
      }
    });

    // Format for human-readable size and time
    const formatBytes = (bytes: number): string => {
      if (bytes === 0) return '0 B';
      const sizes = ['B', 'KB', 'MB', 'GB'];
      const i = Math.floor(Math.log(bytes) / Math.log(1024));
      return `${(bytes / Math.pow(1024, i)).toFixed(2)} ${sizes[i]}`;
    };

    const formatTime = (seconds: number | null): string => {
      if (!seconds || seconds === 0) return '';
      if (seconds < 60) return `${Math.ceil(seconds)}s`;
      if (seconds < 3600) {
        const minutes = Math.floor(seconds / 60);
        const secs = Math.ceil(seconds % 60);
        return `${minutes}m ${secs}s`;
      }
      const hours = Math.floor(seconds / 3600);
      const minutes = Math.floor((seconds % 3600) / 60);
      return `${hours}h ${minutes}m`;
    };

    return {
      progress: overallProgress,
      percentComplete,
      bytesLoaded: this.loadedBytes,
      bytesTotal: this.totalBytes,
      activeFileCount: activeFiles.length,
      totalFileCount: files.length,
      speed: overallSpeed, // bytes per second
      timeRemaining: maxTimeRemaining, // most conservative estimate (longest file)

      // Human readable formats
      formattedLoaded: formatBytes(this.loadedBytes),
      formattedTotal: formatBytes(this.totalBytes),
      formattedSpeed: formatBytes(overallSpeed) + '/s',
      formattedRemaining: formatTime(maxTimeRemaining),

      // Status flags
      isComplete: activeFiles.length === 0 && files.length > 0,
      hasError: files.some(file => file.status === 'error')
    };
  }

  /**
   * Reset tracker state
   */
  reset(): void {
    this.files.clear();
    this.totalBytes = 0;
    this.loadedBytes = 0;
  }
}
