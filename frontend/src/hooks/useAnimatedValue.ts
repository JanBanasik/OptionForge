import { useEffect, useState, useRef } from "react";

/**
 * Animates a numeric value from 0 (or previous value) to target
 * using requestAnimationFrame for smooth updates.
 */
export function useAnimatedValue(target: number, duration = 600, enabled = true) {
  const [displayed, setDisplayed] = useState(target);
  const frameRef = useRef<number>(0);
  const startRef = useRef<number>(0);
  const fromRef = useRef<number>(0);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => { mountedRef.current = false; };
  }, []);

  useEffect(() => {
    if (!enabled) {
      setDisplayed(target);
      return;
    }

    const from = fromRef.current;
    fromRef.current = target;

    // If this is the first render, start from target (no animation on mount)
    if (from === 0 && target > 0) {
      fromRef.current = target;
      setDisplayed(target);
      return;
    }

    // Animate if value changed
    if (Math.abs(target - from) > 0.0001) {
      cancelAnimationFrame(frameRef.current);
      startRef.current = performance.now();

      const animate = (now: number) => {
        const elapsed = now - startRef.current;
        const progress = Math.min(elapsed / duration, 1);
        // ease-out cubic
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = from + (target - from) * eased;

        if (mountedRef.current) {
          setDisplayed(current);
        }

        if (progress < 1) {
          frameRef.current = requestAnimationFrame(animate);
        }
      };

      frameRef.current = requestAnimationFrame(animate);
    }

    return () => cancelAnimationFrame(frameRef.current);
  }, [target, duration, enabled]);

  return displayed;
}
