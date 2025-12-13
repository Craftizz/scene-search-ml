import React from "react";

export function useContainerWidth(
  thumbRef: React.RefObject<HTMLDivElement | null>,
  progressRef: React.RefObject<HTMLDivElement | null>
) {
  const [containerWidth, setContainerWidth] = React.useState(0);

  React.useEffect(() => {
    const update = () => {
      const el = thumbRef.current ?? progressRef.current;
      setContainerWidth(el?.clientWidth ?? 0);
    };
    
    update();
    
    const resizeObserver = new ResizeObserver(update);
    if (thumbRef.current ?? progressRef.current) {
      resizeObserver.observe((thumbRef.current ?? progressRef.current)!);
    }
    
    return () => resizeObserver.disconnect();
  }, [thumbRef, progressRef]);

  return containerWidth;
}
