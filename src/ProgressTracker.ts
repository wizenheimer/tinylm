/**
 * ProgressTracker - Tracks progress with throttling and significance detection
 */

import { ProgressUpdate, ProgressTrackerOptions } from './types';

/**
 * Progress tracker with throttling and percentage support
 */
export class ProgressTracker {
  private callback: (progress: ProgressUpdate) => void;
  private options: Required<ProgressTrackerOptions>;
  private lastUpdateTime: number = 0;
  private lastProgress: ProgressUpdate = {
    status: undefined,
    progress: undefined,
    percentComplete: undefined,
    message: undefined,
    type: undefined
  };

  /**
   * Create a new progress tracker
   * @param {Function} callback - Progress callback function
   * @param {ProgressTrackerOptions} options - Throttling options
   */
  constructor(
    callback?: (progress: ProgressUpdate) => void,
    options: ProgressTrackerOptions = {}
  ) {
    this.callback = callback || (() => {});
    this.options = {
      throttleTime: options.throttleTime || 100, // ms between updates
      significantChangeThreshold: options.significantChangeThreshold || 0.01, // 1% change
      ...options
    } as Required<ProgressTrackerOptions>;
  }

  /**
   * Update progress with throttling
   * @param {ProgressUpdate} progress - Progress object
   */
  update(progress: ProgressUpdate): void {
    try {
      // Sanitize progress value
      const sanitizedProgress: ProgressUpdate = { ...progress };

      // Ensure progress is between 0-1 if provided
      if (typeof progress.progress === 'number') {
        sanitizedProgress.progress = Math.max(0, Math.min(1, progress.progress));
      }

      // Ensure percentComplete is between 0-100 if provided
      if (typeof progress.percentComplete === 'number') {
        sanitizedProgress.percentComplete = Math.max(0, Math.min(100,
          Math.round(progress.percentComplete)));
      } else if (typeof progress.progress === 'number') {
        // Convert progress to percentage if not provided
        sanitizedProgress.percentComplete = Math.round((progress.progress || 0) * 100);
      }

      const now = Date.now();
      const timeSinceLastUpdate = now - this.lastUpdateTime;

      // Determine if update is significant
      const isSignificant = this._isSignificantChange(sanitizedProgress);
      const isStatusChange = sanitizedProgress.status !== this.lastProgress.status;
      const isFirstUpdate = this.lastUpdateTime === 0;
      const isLastUpdate = [
        'ready', 'done', 'error', 'complete', 'interrupted'
      ].includes(sanitizedProgress.status || '');

      if (
        isFirstUpdate ||
        isLastUpdate ||
        isStatusChange ||
        (isSignificant && timeSinceLastUpdate >= this.options.throttleTime)
      ) {
        this.callback(sanitizedProgress);
        this.lastUpdateTime = now;
        this.lastProgress = { ...sanitizedProgress };
      }
    } catch (error) {
      console.error("Error in progress tracker:", error);
    }
  }

  /**
   * Check if a progress update represents a significant change
   * @private
   * @param {ProgressUpdate} progress - Progress object
   * @returns {boolean} True if change is significant
   */
  private _isSignificantChange(progress: ProgressUpdate): boolean {
    if (progress.status !== this.lastProgress.status) return true;
    if (progress.type !== this.lastProgress.type) return true;
    if (progress.message !== this.lastProgress.message) return true;

    // Check either percentage or progress for significance
    if (
      typeof progress.percentComplete === 'number' &&
      typeof this.lastProgress.percentComplete === 'number'
    ) {
      const percentDiff = Math.abs(progress.percentComplete - this.lastProgress.percentComplete);
      return percentDiff >= 1; // 1% change is significant
    } else if (
      typeof progress.progress === 'number' &&
      typeof this.lastProgress.progress === 'number'
    ) {
      const progressDiff = Math.abs(progress.progress - this.lastProgress.progress);
      return progressDiff >= this.options.significantChangeThreshold;
    }

    return false;
  }
}
