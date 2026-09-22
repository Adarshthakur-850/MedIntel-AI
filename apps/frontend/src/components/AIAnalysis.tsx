import React from 'react';
import { Cpu, Layers, Image as ImageIcon, FileText, Zap } from 'lucide-react';

export interface AnalysisResults {
  patient_id: string;
  tabular_risk: number;
  image_risk: number;
  nlp_risk: number;
  fused_risk: number;
  fusion_strategy: string;
  disclaimer: string;
}

interface Props {
  results: AnalysisResults | null;
}

export const AIAnalysis: React.FC<Props> = ({ results }) => {
  if (!results) {
    return (
      <div className="bg-slate-800 border border-slate-700 rounded-xl p-8 text-center text-slate-400">
        <Cpu className="w-12 h-12 mx-auto text-slate-600 mb-3" />
        <p className="text-base font-medium">No active analysis results available.</p>
        <p className="text-xs text-slate-500 mt-1">Submit patient data above to trigger multimodal risk fusion.</p>
      </div>
    );
  }

  const getRiskBadge = (prob: number) => {
    if (prob >= 0.75) return <span className="px-2.5 py-1 bg-rose-500/10 border border-rose-500/30 text-rose-400 font-semibold rounded-full text-xs">High Risk ({ (prob * 100).toFixed(1) }%)</span>;
    if (prob >= 0.40) return <span className="px-2.5 py-1 bg-amber-500/10 border border-amber-500/30 text-amber-400 font-semibold rounded-full text-xs">Moderate Risk ({ (prob * 100).toFixed(1) }%)</span>;
    return <span className="px-2.5 py-1 bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 font-semibold rounded-full text-xs">Low Risk ({ (prob * 100).toFixed(1) }%)</span>;
  };

  return (
    <div className="space-y-6">
      {/* Fused Risk Metric Header */}
      <div className="bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700 rounded-xl p-6 shadow-xl relative overflow-hidden">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-3">
            <Zap className="w-6 h-6 text-amber-400" />
            <div>
              <h3 className="text-lg font-semibold text-white">Fused Multimodal AI Risk</h3>
              <p className="text-xs text-slate-400">Strategy: {results.fusion_strategy === 'neural' ? 'Learned Neural Network Fusion' : 'Weighted Late Consensus'}</p>
            </div>
          </div>
          {getRiskBadge(results.fused_risk)}
        </div>

        {/* Progress Bar */}
        <div className="w-full bg-slate-700 rounded-full h-3.5 mb-3 overflow-hidden">
          <div
            className={`h-3.5 rounded-full transition-all duration-500 ${
              results.fused_risk >= 0.75 ? 'bg-gradient-to-r from-amber-500 to-rose-500' :
              results.fused_risk >= 0.40 ? 'bg-gradient-to-r from-sky-500 to-amber-500' : 'bg-gradient-to-r from-emerald-500 to-sky-500'
            }`}
            style={{ width: `${Math.min(results.fused_risk * 100, 100)}%` }}
          />
        </div>
      </div>

      {/* Individual Modality Breakdown Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        <div className="bg-slate-800 border border-slate-700 rounded-xl p-5">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center space-x-2">
              <Layers className="w-5 h-5 text-sky-400" />
              <h4 className="text-sm font-semibold text-slate-200">Clinical Tabular</h4>
            </div>
            {getRiskBadge(results.tabular_risk)}
          </div>
          <p className="text-xs text-slate-400">Vitals, Blood Glucose, Creatinine, BP, Age.</p>
        </div>

        <div className="bg-slate-800 border border-slate-700 rounded-xl p-5">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center space-x-2">
              <ImageIcon className="w-5 h-5 text-indigo-400" />
              <h4 className="text-sm font-semibold text-slate-200">Chest X-Ray Vision</h4>
            </div>
            {getRiskBadge(results.image_risk)}
          </div>
          <p className="text-xs text-slate-400">PyTorch ResNet18 opacity & pulmonary scan.</p>
        </div>

        <div className="bg-slate-800 border border-slate-700 rounded-xl p-5">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center space-x-2">
              <FileText className="w-5 h-5 text-emerald-400" />
              <h4 className="text-sm font-semibold text-slate-200">Clinical Text NLP</h4>
            </div>
            {getRiskBadge(results.nlp_risk)}
          </div>
          <p className="text-xs text-slate-400">Clinical note sentiment & NER extraction.</p>
        </div>
      </div>
    </div>
  );
};
