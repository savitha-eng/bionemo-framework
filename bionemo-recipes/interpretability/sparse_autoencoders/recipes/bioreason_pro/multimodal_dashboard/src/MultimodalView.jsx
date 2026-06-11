// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-Apache2
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import React, { useEffect, useRef } from 'react'
import * as vg from '@uwdata/vgplot'

// Multimodal feature view for BioReason-Pro: position each SAE feature by its BAND COMPOSITION
// (fraction of its activation RATE that lands in the protein vs text band; go is the remainder),
// color by band_class, and RING the truly cross-modal features (fusion_class != 'unimodal' = a
// text-involving SAE-V fusion). So protein-heavy features sit top-left, text-heavy bottom-right,
// go-heavy bottom-left, and a ringed point = a feature whose protein/go and text activations point
// the same direction (the model represents that concept across modalities).
//
// Reads the `features` DuckDB table already loaded by App.jsx via the mosaic coordinator.
export default function MultimodalView({ onFeatureClick, highlightedFeatureId, darkMode }) {
  const containerRef = useRef(null)
  const clickRef = useRef(onFeatureClick)
  clickRef.current = onFeatureClick

  useEffect(() => {
    if (!containerRef.current) return
    let plot = null
    const axis = darkMode ? '#888' : '#666'

    try {
      plot = vg.plot(
        // all features, colored by composition class
        vg.dot(vg.from('features'), {
          x: 'text_frac', y: 'protein_frac',
          fill: 'band_class',
          r: 2.5, fillOpacity: 0.55,
        }),
        // truly cross-modal features: larger ring on top
        vg.dot(vg.from('features', { filter: vg.sql`fusion_class <> 'unimodal'` }), {
          x: 'text_frac', y: 'protein_frac',
          r: 6, fill: 'none', stroke: '#76B900', strokeWidth: 2,
          tip: true,
          channels: { feature: 'feature_id', concept: 'go_label', omega: 'crossmodal_omega', fusion: 'fusion_class' },
        }),
        // hover detail on the full cloud too
        vg.dot(vg.from('features'), {
          x: 'text_frac', y: 'protein_frac', r: 2.5, fill: 'none',
          tip: true,
          channels: { feature: 'feature_id', band: 'band_class', concept: 'go_label', omega: 'crossmodal_omega' },
        }),
        vg.colorDomain(['text-heavy', 'protein-heavy', 'go-heavy', 'protein-go-shared', 'mixed']),
        vg.colorRange(['#0074DF', '#EF2020', '#76B900', '#ef9100', '#9aa0a6']),
        vg.colorLegend({ label: 'composition' }),
        vg.xDomain([0, 1]), vg.yDomain([0, 1]),
        vg.xLabel('text fraction →'), vg.yLabel('protein fraction →'),
        vg.width(560), vg.height(460),
        vg.style({ color: axis, fontSize: '12px', background: 'transparent' }),
        vg.margins({ left: 50, bottom: 45, top: 10, right: 10 }),
      )
      containerRef.current.replaceChildren(plot)

      // click-to-select: nearest feature to the click point
      const svg = containerRef.current.querySelector('svg')
      if (svg) {
        svg.style.cursor = 'crosshair'
        svg.addEventListener('click', async (ev) => {
          const rect = svg.getBoundingClientRect()
          // map pixel -> data is non-trivial without scales; query nearest via DuckDB on the
          // normalized [0,1] coords using the plot's frame (approximate, good enough for drill-in).
          const fx = (ev.clientX - rect.left - 60) / (rect.width - 70)
          const fy = 1 - (ev.clientY - rect.top - 10) / (rect.height - 55)
          if (fx < 0 || fx > 1 || fy < 0 || fy > 1) return
          try {
            const res = await vg.coordinator().query(
              `SELECT feature_id FROM features
               ORDER BY (text_frac-${fx})*(text_frac-${fx}) + (protein_frac-${fy})*(protein_frac-${fy})
               LIMIT 1`)
            const row = Array.from(res)[0]
            if (row && clickRef.current) clickRef.current(Number(row.feature_id))
          } catch (e) { /* ignore */ }
        })
      }
    } catch (err) {
      containerRef.current.textContent = `Multimodal view error: ${err.message}`
    }

    return () => { if (plot && plot.remove) plot.remove() }
  }, [darkMode, highlightedFeatureId])

  return <div ref={containerRef} style={{ width: '100%', height: '100%', overflow: 'auto' }} />
}
