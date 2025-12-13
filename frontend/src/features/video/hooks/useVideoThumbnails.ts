import React from "react";

interface Frame {
  timestamp: number;
  url: string;
  caption?: string;
  embedding?: number[];
}

export function useVideoThumbnails(
  frames: any[] | undefined,
  containerWidth: number,
  duration: number | null
) {
  // Pre-calculate target timestamps based on duration - NEVER recalculates
  const targetTimestamps = React.useMemo(() => {
    const THUMB_WIDTH = 48;
    const MIN_THUMBS = 2;
    const MAX_THUMBS = 12;

    if (!containerWidth || !duration || duration <= 0) {
      return [];
    }

    const maxPossible = Math.floor(containerWidth / THUMB_WIDTH);
    const count = Math.max(MIN_THUMBS, Math.min(MAX_THUMBS, maxPossible || 6));
    const interval = duration / count;

    // Calculate which timestamps we want - this is fixed and never changes
    return Array.from({ length: count }, (_, i) => ({
      target: Math.round((i + 0.5) * interval),
      tolerance: Math.ceil(interval * 0.5)
    }));
  }, [containerWidth, duration]);

  // Stateful slots that only update when filled - prevents constant recalculation
  const [thumbnails, setThumbnails] = React.useState<(Frame | null)[]>([]);
  const filledSlotsRef = React.useRef<Set<number>>(new Set());
  
  // Reset slots when target timestamps change
  React.useEffect(() => {
    if (targetTimestamps.length === 0) {
      setThumbnails([]);
      filledSlotsRef.current.clear();
      return;
    }
    
    // Initialize with nulls
    setThumbnails(new Array(targetTimestamps.length).fill(null));
    filledSlotsRef.current.clear();
  }, [targetTimestamps]);

  // Update slots as frames arrive - only touches unfilled slots
  React.useEffect(() => {
    if (targetTimestamps.length === 0 || !frames) return;

    // Build map of available finalized frames
    const frameMap = new Map<number, Frame>();
    (frames || []).forEach((f: any) => {
      if ((f.embedding?.length > 0) || (f.caption?.length > 0)) {
        const ts = Math.round(Number(f.timestamp) || 0);
        if (!frameMap.has(ts)) {
          frameMap.set(ts, {
            timestamp: ts,
            url: f.url,
            caption: f.caption,
            embedding: f.embedding
          });
        }
      }
    });

    // Check unfilled slots and fill them if frame is available
    const updates: { index: number; frame: Frame }[] = [];
    
    targetTimestamps.forEach(({ target, tolerance }, index) => {
      // Skip already filled slots - NEVER replace
      if (filledSlotsRef.current.has(index)) return;

      // Check exact match first
      let matchedFrame = frameMap.get(target);
      
      // Otherwise spiral outward
      if (!matchedFrame) {
        for (let offset = 1; offset <= tolerance; offset++) {
          const earlier = frameMap.get(target - offset);
          if (earlier) {
            matchedFrame = earlier;
            break;
          }
          
          const later = frameMap.get(target + offset);
          if (later) {
            matchedFrame = later;
            break;
          }
        }
      }

      if (matchedFrame) {
        updates.push({ index, frame: matchedFrame });
        filledSlotsRef.current.add(index);
      }
    });

    // Only update state if we have new frames to add
    if (updates.length > 0) {
      setThumbnails(prev => {
        const next = [...prev];
        updates.forEach(({ index, frame }) => {
          next[index] = frame;
        });
        return next;
      });
    }
  }, [frames, targetTimestamps]);

  return thumbnails;
}
