import { AsyncLocalStorage } from '@openai/agents-core/_shims';
import { Trace, TraceOptions } from './traces';
import { getGlobalTraceProvider } from './provider';
import { Span, SpanError } from './spans';
import { StreamedRunResult } from '../result';

type ContextState = {
  trace?: Trace;
  span?: Span<any>;
  previousSpan?: Span<any>;
  active?: boolean;
};

const ALS_SYMBOL = Symbol.for('openai.agents.core.asyncLocalStorage');
const CONTEXT_SYMBOL = Symbol.for('openai.agents.core.lastContext');
let localFallbackAls: AsyncLocalStorage<ContextState> | undefined;

// Global symbols ensure that if multiple copies of agents-core are loaded
// (e.g., via different npm resolution paths or bundlers), they all share the
// same AsyncLocalStorage instance and last-known context. This prevents losing
// trace/span state when a downstream package pulls in a duplicate copy.
// The global fallback should be considered a best-effort safety net only; the
// primary isolation still comes from AsyncLocalStorage when available.
function getContextAsyncLocalStorage() {
  try {
    const globalScope = globalThis as unknown as Record<
      symbol | string,
      AsyncLocalStorage<ContextState> | undefined
    >;

    const globalALS = globalScope[ALS_SYMBOL];

    if (globalALS) {
      return globalALS;
    }

    const newALS = new AsyncLocalStorage<ContextState>();
    globalScope[ALS_SYMBOL] = newALS;
    return newALS;
  } catch {
    // As a defensive fallback (e.g., if globalThis is locked down or ALS
    // construction throws in a constrained runtime), keep a module-local ALS
    // so tracing still functions instead of crashing callers.
    if (!localFallbackAls) {
      localFallbackAls = new AsyncLocalStorage<ContextState>();
    }
    return localFallbackAls;
  }
}

// Store the latest context in globalThis so that, if AsyncLocalStorage store
// lookup fails (duplicate copy, boundary hops), we can still resume tracing.
function setGlobalContext(context: ContextState) {
  const globalScope = globalThis as unknown as Record<
    symbol | string,
    ContextState | undefined
  >;
  globalScope[CONTEXT_SYMBOL] = context;
}

// Retrieve the fallback context if AsyncLocalStorage has no store. This is
// a best-effort safety net for environments that accidentally load multiple
// copies of agents-core or lose ALS scope (e.g., certain worker runtimes).
function getGlobalContext(): ContextState | undefined {
  const globalScope = globalThis as unknown as Record<
    symbol | string,
    ContextState | undefined
  >;
  return globalScope[CONTEXT_SYMBOL];
}

function restoreGlobalContext(
  expectedContext: ContextState,
  previousContext?: ContextState,
) {
  const globalScope = globalThis as unknown as Record<
    symbol | string,
    ContextState | undefined
  >;

  // Only restore if the global fallback still points to the context this trace
  // installed. If another concurrent trace updated the global context in the
  // meantime, leave it intact to avoid clobbering that run.
  if (globalScope[CONTEXT_SYMBOL] !== expectedContext) {
    return;
  }

  if (previousContext?.active) {
    globalScope[CONTEXT_SYMBOL] = previousContext;
  } else {
    delete globalScope[CONTEXT_SYMBOL];
  }
}

/**
 * This function will get the current trace from the execution context.
 *
 * @returns The current trace or null if there is no trace.
 */
export function getCurrentTrace() {
  const currentTrace =
    getContextAsyncLocalStorage().getStore() ?? getGlobalContext();
  if (currentTrace?.trace) {
    return currentTrace.trace;
  }

  return null;
}

/**
 * This function will get the current span from the execution context.
 *
 * @returns The current span or null if there is no span.
 */
export function getCurrentSpan() {
  const currentSpan =
    getContextAsyncLocalStorage().getStore() ?? getGlobalContext();
  if (currentSpan?.span) {
    return currentSpan.span;
  }
  return null;
}

/**
 * This is an AsyncLocalStorage instance that stores the current trace.
 * It will automatically handle the execution context of different event loop executions.
 *
 * The functions below should be the only way that this context gets interfaced with.
 */
function _wrapFunctionWithTraceLifecycle<T>(
  fn: (trace: Trace) => Promise<T>,
  currentContext: ContextState,
  previousContext?: ContextState,
) {
  return async () => {
    const trace = getCurrentTrace();
    if (!trace) {
      throw new Error('No trace found');
    }

    await trace.start();
    let cleanupDeferred = false;

    try {
      const result = await fn(trace);

      // If result is a StreamedRunResult, defer trace end until stream loop completes
      if (result instanceof StreamedRunResult) {
        const streamLoopPromise = result._getStreamLoopPromise();
        if (streamLoopPromise) {
          cleanupDeferred = true;
          streamLoopPromise.finally(async () => {
            await trace.end();

            currentContext.active = false;
            restoreGlobalContext(currentContext, previousContext);
          });

          return result;
        }
      }

      // For non-streaming results, end trace synchronously
      await trace.end();

      return result;
    } finally {
      currentContext.active = false;
      // Always restore prior global context (or clear) so parallel callers do
      // not see stale span/trace state if they share the same process.
      if (!cleanupDeferred) {
        restoreGlobalContext(currentContext, previousContext);
      }
    }
  };
}

/**
 * This function will create a new trace and assign it to the execution context of the function
 * passed to it.
 *
 * @param fn - The function to run and assign the trace context to.
 * @param options - Options for the creation of the trace
 */

export async function withTrace<T>(
  trace: string | Trace,
  fn: (trace: Trace) => Promise<T>,
  options: TraceOptions = {},
): Promise<T> {
  const newTrace =
    typeof trace === 'string'
      ? getGlobalTraceProvider().createTrace({
          ...options,
          name: trace,
        })
      : trace;

  const context: ContextState = { trace: newTrace, active: true };
  const previousContext = getGlobalContext();
  setGlobalContext(context);

  return getContextAsyncLocalStorage().run(
    context,
    _wrapFunctionWithTraceLifecycle(fn, context, previousContext),
  );
}
/**
 * This function will check if there is an existing active trace in the execution context. If there
 * is, it will run the given function with the existing trace. If there is no trace, it will create
 * a new one and assign it to the execution context of the function.
 *
 * @param fn - The fzunction to run and assign the trace context to.
 * @param options - Options for the creation of the trace
 */
export async function getOrCreateTrace<T>(
  fn: () => Promise<T>,
  options: TraceOptions = {},
): Promise<T> {
  const currentTrace = getCurrentTrace();
  if (currentTrace) {
    // if this execution context already has a trace instance in it we just continue
    const existingContext =
      getContextAsyncLocalStorage().getStore() ?? getGlobalContext();
    if (existingContext) {
      setGlobalContext(existingContext);
      getContextAsyncLocalStorage().enterWith(existingContext);
    }
    return await fn();
  }

  const newTrace = getGlobalTraceProvider().createTrace(options);

  const newContext: ContextState = { trace: newTrace, active: true };
  const previousContext = getGlobalContext();
  setGlobalContext(newContext);
  return getContextAsyncLocalStorage().run(
    newContext,
    _wrapFunctionWithTraceLifecycle(fn, newContext, previousContext),
  );
}

/**
 * This function will set the current span in the execution context.
 *
 * @param span - The span to set as the current span.
 */
export function setCurrentSpan(span: Span<any>) {
  const context =
    getContextAsyncLocalStorage().getStore() ?? getGlobalContext();
  if (!context) {
    throw new Error('No existing trace found');
  }

  if (context.span) {
    context.span.previousSpan = context.previousSpan;
    context.previousSpan = context.span;
  }

  span.previousSpan = context.span ?? context.previousSpan;
  context.span = span;
  getContextAsyncLocalStorage().enterWith(context);
  setGlobalContext(context);
}

export function resetCurrentSpan() {
  const context =
    getContextAsyncLocalStorage().getStore() ?? getGlobalContext();
  if (context) {
    context.span = context.previousSpan;
    context.previousSpan = context.previousSpan?.previousSpan;
    getContextAsyncLocalStorage().enterWith(context);
    setGlobalContext(context);
  }
}

/**
 * This function will add an error to the current span.
 *
 * @param spanError - The error to add to the current span.
 */
export function addErrorToCurrentSpan(spanError: SpanError) {
  const currentSpan = getCurrentSpan();
  if (currentSpan) {
    currentSpan.setError(spanError);
  }
}

/**
 * This function will clone the current context by creating new instances of the trace, span, and
 * previous span.
 *
 * @param context - The context to clone.
 * @returns A clone of the context.
 */
export function cloneCurrentContext(context: ContextState) {
  return {
    trace: context.trace?.clone(),
    span: context.span?.clone(),
    previousSpan: context.previousSpan?.clone(),
  };
}

/**
 * This function will run the given function with a new span context.
 *
 * @param fn - The function to run with the new span context.
 */
export function withNewSpanContext<T>(fn: () => Promise<T>) {
  const currentContext =
    getContextAsyncLocalStorage().getStore() ?? getGlobalContext();
  if (!currentContext) {
    return fn();
  }

  const copyOfContext = cloneCurrentContext(currentContext);

  return getContextAsyncLocalStorage().run(copyOfContext, fn);
}
