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
  private _overallProgress: OverallProgress = {
    progress: 0,
    percentComplete: 0,
    bytesLoaded: 0,
    bytesTotal: 0,
    activeFileCount: 0,
    totalFileCount: 0,
    speed: 0,
    timeRemaining: 0,
    formattedLoaded: '0 B',
    formattedTotal: '0 B',
    formattedSpeed: '0 B/s',
    formattedRemaining: '0s',
    isComplete: false,
    hasError: false
  };

  constructor() {
    this.totalBytes = 0;
    this.loadedBytes = 0;
  }

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
   * Update progress for a file
   */
  update(progress: any): void {
    const fileId = progress.file || progress.url || Date.now().toString();
    const file = this.getOrCreateFile(fileId);

    // Update file progress
    file.progress = progress.progress || 0;
    file.percentComplete = progress.progress ? Math.round(progress.progress * 100) : 0;
    file.bytesLoaded = progress.loaded || 0;
    file.bytesTotal = progress.total || 0;
    file.lastUpdateTime = Date.now();

    // Calculate speed and time remaining
    const timeDiff = (file.lastUpdateTime - file.startTime) / 1000; // in seconds
    if (timeDiff > 0) {
      file.speed = file.bytesLoaded / timeDiff;
      if (file.speed > 0) {
        const bytesRemaining = file.bytesTotal - file.bytesLoaded;
        file.timeRemaining = bytesRemaining / file.speed;
      }
    }

    // Update overall progress
    this.updateOverallProgress();
  }

  /**
   * Get all tracked files
   */
  getAllFiles(): FileInfo[] {
    return Array.from(this.files.values());
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
   * Get active (incomplete) files
   * @returns {FileInfo[]} Array of active file information objects
   */
  getActiveFiles(): FileInfo[] {
    return this.getAllFiles().filter(file =>
      file.status !== 'done' && file.status !== 'error');
  }

  /**
   * Get overall progress
   */
  getOverallProgress(): OverallProgress {
    return this._overallProgress;
  }

  /**
   * Reset tracker state
   */
  reset(): void {
    this.files.clear();
    this.totalBytes = 0;
    this.loadedBytes = 0;
  }

  /**
   * Get or create a file entry
   */
  private getOrCreateFile(fileId: string): FileInfo {
    if (!this.files.has(fileId)) {
      const now = Date.now();
      const file: FileInfo = {
        id: fileId,
        name: fileId,
        status: 'loading',
        progress: 0,
        percentComplete: 0,
        bytesLoaded: 0,
        bytesTotal: 0,
        startTime: now,
        lastUpdateTime: now,
        speed: 0,
        timeRemaining: null
      };
      this.files.set(fileId, file);
      return file;
    }
    return this.files.get(fileId)!;
  }

  /**
   * Update overall progress
   */
  private updateOverallProgress(): void {
    let totalBytes = 0;
    let loadedBytes = 0;
    let activeFiles = 0;
    let totalFiles = this.files.size;
    let maxSpeed = 0;
    let maxTimeRemaining = 0;

    for (const file of this.files.values()) {
      if (file.bytesTotal > 0) {
        totalBytes += file.bytesTotal;
        loadedBytes += file.bytesLoaded;
      }
      if (file.status === 'loading') {
        activeFiles++;
        if (file.speed > maxSpeed) maxSpeed = file.speed;
        if (file.timeRemaining && file.timeRemaining > maxTimeRemaining) {
          maxTimeRemaining = file.timeRemaining;
        }
      }
    }

    this._overallProgress = {
      progress: totalBytes > 0 ? loadedBytes / totalBytes : 0,
      percentComplete: totalBytes > 0 ? Math.round((loadedBytes / totalBytes) * 100) : 0,
      bytesLoaded: loadedBytes,
      bytesTotal: totalBytes,
      activeFileCount: activeFiles,
      totalFileCount: totalFiles,
      speed: maxSpeed,
      timeRemaining: maxTimeRemaining,
      formattedLoaded: this.formatBytes(loadedBytes),
      formattedTotal: this.formatBytes(totalBytes),
      formattedSpeed: this.formatSpeed(maxSpeed),
      formattedRemaining: this.formatTime(maxTimeRemaining),
      isComplete: activeFiles === 0 && totalFiles > 0,
      hasError: false
    };
  }

  /**
   * Format bytes to human readable string
   */
  private formatBytes(bytes: number): string {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${(bytes / Math.pow(k, i)).toFixed(2)} ${sizes[i]}`;
  }

  /**
   * Format speed to human readable string
   */
  private formatSpeed(bytesPerSecond: number): string {
    if (bytesPerSecond === 0) return '0 B/s';
    return `${this.formatBytes(bytesPerSecond)}/s`;
  }

  /**
   * Format time to human readable string
   */
  private formatTime(seconds: number): string {
    if (!seconds || seconds === 0) return '0s';
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.round(seconds % 60);
    const parts = [];
    if (h > 0) parts.push(`${h}h`);
    if (m > 0) parts.push(`${m}m`);
    if (s > 0) parts.push(`${s}s`);
    return parts.join(' ');
  }
}
