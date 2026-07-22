import React from 'react';

/**
 * Safion brand logo — the 4×4 grid of rectangles forming the "S" motif.
 *
 * Extracted from LoginPage.jsx and App.js to eliminate the ~140-line
 * duplication.  Pass ``className``, ``width``, and ``height`` to control
 * sizing; defaults to 64×82 (sidebar collapsed size).
 */
export default function SafionLogo({ className, width = 64, height = 82 }) {
  return (
    <svg width={width} height={height} viewBox="0 0 123 152" className={className}>
      <rect x="8"  y="8"   width="20" height="20" rx="8" fill="#181826"/>
      <rect x="37" y="8"   width="20" height="20" rx="8" fill="#F54F00"/>
      <rect x="66" y="8"   width="20" height="20" rx="8" fill="#F54F00"/>
      <rect x="95" y="8"   width="20" height="20" rx="8" fill="#F54F00"/>
      <rect x="8"  y="37"  width="20" height="20" rx="8" fill="#F54F00"/>
      <rect x="37" y="37"  width="20" height="20" rx="8" fill="#181826"/>
      <rect x="66" y="37"  width="20" height="20" rx="8" fill="#181826"/>
      <rect x="95" y="37"  width="20" height="20" rx="8" fill="#181826"/>
      <rect x="8"  y="66"  width="20" height="20" rx="8" fill="#181826"/>
      <rect x="37" y="66"  width="20" height="20" rx="8" fill="#F54F00"/>
      <rect x="66" y="66"  width="20" height="20" rx="8" fill="#F54F00"/>
      <rect x="95" y="66"  width="20" height="20" rx="8" fill="#181826"/>
      <rect x="8"  y="95"  width="20" height="20" rx="8" fill="#181826"/>
      <rect x="37" y="95"  width="20" height="20" rx="8" fill="#181826"/>
      <rect x="66" y="95"  width="20" height="20" rx="8" fill="#181826"/>
      <rect x="95" y="95"  width="20" height="20" rx="8" fill="#F54F00"/>
      <rect x="8"  y="124" width="20" height="20" rx="8" fill="#F54F00"/>
      <rect x="37" y="124" width="20" height="20" rx="8" fill="#F54F00"/>
      <rect x="66" y="124" width="20" height="20" rx="8" fill="#F54F00"/>
      <rect x="95" y="124" width="20" height="20" rx="8" fill="#181826"/>
    </svg>
  );
}