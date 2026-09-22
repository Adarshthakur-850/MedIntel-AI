import React from 'react';
import { Eye, TrendingUp, TrendingDown, Image as ImageIcon, CheckCircle, AlertTriangle } from 'lucide-react';

export interface ExplanationData {
  shap_explanation: {
    top_positive_contributors: Array<{ feature: string; feature_value: number; shap_value: number }>;
    top_negative_contributors: Array<{ feature: string; feature_value: number; shap_value: number }>;
    global_feature_importance: Array<{ feature: string; importance: number }>;
  };
  nlp_explanation: {
    highlighted_phrases: Array<{ phrase: string; risk_weight: number; impact: string }>;
    explanation_summary: string;
  };
  vision_heatmap_path?: string;
}

interface Props {
  explanation: ExplanationData | null;
}

export const ExplainabilityView: React.FC<Props> = ({ explanation }) => {
  if (!explanation) {
    return (
      <div className="bg-slate-800 border border-slate-700 rounded-xl p-8 text-center text-slate-400">
        <Eye className="w-12 h-12 mx-auto text-slate-600 mb-3" />
        <p className="text-base font-medium">Explainability insights pending model inference.</p>
      </div>
    );
  }

  const { shap_explanation, nlp_explanation } = explanation;

  return (
    <div className="space-y-6">
      <div className="bg-sky-950/40 border border-sky-500/30 rounded-xl p-4 flex items-center space-x-3 text-sky-200 text-xs">
        <AlertTriangle className="w-5 h-5 text-sky-400 flex-shrink-0" />
        <p>
          <strong className="font-semibold">Model Behavior Explanation:</strong> Displays local feature contributions (SHAP), image region activations (Grad-CAM), and text evidence. This section explains <em>how</em> the AI model calculated risk, not medical causality.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* SHAP Tabular Feature Attribution */}
        <div className="bg-slate-800 border border-slate-700 rounded-xl p-6 shadow-xl">
          <div className="flex items-center space-x-2 mb-4 pb-3 border-b border-slate-700">
            <TrendingUp className="w-5 h-5 text-amber-400" />
            <h3 className="text-base font-semibold text-white">SHAP Tabular Feature Attribution</h3>
          </div>

          <div className="space-y-4">
            <div>
              <h4 className="text-xs font-semibold text-rose-400 uppercase tracking-wider mb-2">Top Risk-Increasing Features (+)</h4>
              <div className="space-y-2">
                {shap_explanation.top_positive_contributors.map((item, idx) => (
                  <div key={idx} className="flex items-center justify-between bg-slate-900/60 p-2.5 rounded-lg border border-slate-700/50">
                    <span className="text-xs font-medium text-slate-200">{item.feature} ({item.feature_value})</span>
                    <span className="text-xs font-mono font-bold text-rose-400">+{item.shap_value.toFixed(4)}</span>
                  </div>
                ))}
              </div>
            </div>

            <div>
              <h4 className="text-xs font-semibold text-emerald-400 uppercase tracking-wider mb-2">Top Risk-Reducing Features (-)</h4>
              <div className="space-y-2">
                {shap_explanation.top_negative_contributors.map((item, idx) => (
                  <div key={idx} className="flex items-center justify-between bg-slate-900/60 p-2.5 rounded-lg border border-slate-700/50">
                    <span className="text-xs font-medium text-slate-200">{item.feature} ({item.feature_value})</span>
                    <span className="text-xs font-mono font-bold text-emerald-400">{item.shap_value.toFixed(4)}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* NLP Evidence & Grad-CAM Heatmap Preview */}
        <div className="space-y-6">
          <div className="bg-slate-800 border border-slate-700 rounded-xl p-6 shadow-xl">
            <div className="flex items-center space-x-2 mb-4 pb-3 border-b border-slate-700">
              <CheckCircle className="w-5 h-5 text-emerald-400" />
              <h3 className="text-base font-semibold text-white">Clinical Note Key Evidence</h3>
            </div>
            <div className="flex flex-wrap gap-2 mb-3">
              {nlp_explanation.highlighted_phrases.map((h, idx) => (
                <span
                  key={idx}
                  className={`px-3 py-1 text-xs font-medium rounded-full border ${
                    h.risk_weight > 0.6
                      ? 'bg-rose-500/20 border-rose-500/40 text-rose-300'
                      : 'bg-amber-500/20 border-amber-500/40 text-amber-300'
                  }`}
                >
                  {h.phrase} (Weight: {h.risk_weight})
                </span>
              ))}
            </div>
            <p className="text-xs text-slate-400">{nlp_explanation.explanation_summary}</p>
          </div>

          <div className="bg-slate-800 border border-slate-700 rounded-xl p-6 shadow-xl">
            <div className="flex items-center space-x-2 mb-3">
              <ImageIcon className="w-5 h-5 text-indigo-400" />
              <h3 className="text-base font-semibold text-white">Chest X-Ray Grad-CAM Heatmap</h3>
            </div>
            <p className="text-xs text-slate-400 mb-3">
              Highlights chest cavity regions (e.g. lung fields) driving network opacity classification.
            </p>
            <div className="w-full h-40 bg-slate-900 rounded-lg border border-slate-700 flex items-center justify-center text-slate-500 text-xs font-mono">
              [ Grad-CAM Heatmap Activation Map Generated ]
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
