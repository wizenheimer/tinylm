import { StoppingCriteriaList } from "./types";

// Update InterruptableStoppingCriteria to implement StoppingCriteriaList
export class InterruptableStoppingCriteria implements StoppingCriteriaList {
  criteria: any[] = [];
  private interrupted: boolean = false;

  push(criterion: any): void {
    this.criteria.push(criterion);
  }

  extend(criteria: any[]): void {
    this.criteria.push(...criteria);
  }

  [Symbol.iterator](): IterableIterator<any> {
    return this.criteria[Symbol.iterator]();
  }

  shouldStop(input_ids: number[][], scores: number[][], options?: any): boolean {
    return this.interrupted;
  }

  interrupt(): void {
    this.interrupted = true;
  }

  reset(): void {
    this.interrupted = false;
  }

  /**
   * Create a proxy that makes this object callable for transformers.js
   */
  asCallable(): any {
    const self = this;
    return new Proxy(this, {
      apply(_target: any, _thisArg: any, args: [number[][], number[][], ...any[]]): any[] {
        return [self.shouldStop(args[0], args[1], args[2])];
      },
      get(target: any, prop: string | symbol): any {
        if (prop === Symbol.iterator) {
          return target[Symbol.iterator].bind(target);
        }
        return target[prop];
      }
    });
  }
}


/**
 * Interruptable generation with the ability to stop generation midway
 */
export class GenerationController {
  stoppingCriteria: InterruptableStoppingCriteria;
  isGenerating: boolean = false;
  private pastKeyValuesCache: any = null;

  constructor() {
    this.stoppingCriteria = new InterruptableStoppingCriteria();
  }

  /**
   * Interrupt the current generation
   * @returns {boolean} Whether generation was successfully interrupted
   */
  interrupt(): boolean {
    if (this.isGenerating) {
      this.stoppingCriteria.interrupt();
      this.isGenerating = false;
      return true;
    }
    return false;
  }

  /**
   * Reset the controller state
   */
  reset(): void {
    this.stoppingCriteria.reset();
    this.pastKeyValuesCache = null;
    this.isGenerating = false;
  }

  /**
   * Get the stopping criteria for generation
   * @returns {any} Stopping criteria object
   */
  getStoppingCriteria(): any {
    return this.stoppingCriteria.asCallable();
  }

  /**
   * Get cached key values from previous generation
   * @returns {any|null} Past key values or null if not available
   */
  getPastKeyValues(): any {
    return this.pastKeyValuesCache;
  }

  /**
   * Set cached key values from generation
   * @param {any} pastKeyValues - Key values to cache
   */
  setPastKeyValues(pastKeyValues: any): void {
    this.pastKeyValuesCache = pastKeyValues;
  }
}

