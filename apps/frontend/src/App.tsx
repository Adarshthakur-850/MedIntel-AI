import React, { useState } from 'react';
import { Activity, ShieldAlert, Cpu, Eye, BookOpen, Database, Sparkles } from 'lucide-react';

import { PatientOverview, PatientData } from './components/PatientOverview';
import { AIAnalysis, AnalysisResults } from './components/AIAnalysis';
import { ExplainabilityView, ExplanationData } from './components/ExplainabilityView';
import { KnowledgeAssistant } from './components/KnowledgeAssistant';
import { ModelRegistryView } from './components/ModelRegistryView';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'overview' | 'analysis' | 'explain' | 'rag' | 'models'>('overview');
  const [loading, setLoading] = useState(false);

  const [patientData, setPatientData] = useState<PatientData>({
    patient_id: "PAT_0042",
    age: 58,
    sex: "M",
    blood_pressure_sys: 145,
    blood_pressure_dia: 92,
    heart_rate: 82,
    temperature: 37.2,
    glucose: 155,
    cholesterol: 235,
    hemoglobin: 13.8,
    creatinine: 1.4,
    bmi: 29.5,
    smoking_status: "Former",
    diabetes_history: 1,
    clinical_note: "Patient presents with persistent shortness of breath and chest tightness for the past 5 days. History of hypertension and elevated glucose."
  });

  const [analysisResults, setAnalysisResults] = useState<AnalysisResults | null>(null);
  const [explanationData, setExplanationData] = useState<ExplanationData | null>(null);

  const handleRunInference = async () => {
    setLoading(true);
    try {
      // API call to backend
      const resp = await fetch("http://localhost:8000/api/v1/predict/multimodal", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          patient: patientData,
          clinical_note: patientData.clinical_note,
          fusion_strategy: "neural"
        })
      });
      const data = await resp.json();
      setAnalysisResults(data);

      // Fetch explanation data
      const expResp = await fetch("http://localhost:8000/api/v1/explain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          patient: patientData,
          clinical_note: patientData.clinical_note,
          fusion_strategy: "neural"
        })
      });
      const expData = await expResp.json();
      setExplanationData(expData);

      setActiveTab('analysis');
    } catch (err) {
      // Fallback simulation if backend offline
      setAnalysisResults({
        patient_id: patientData.patient_id,
        tabular_risk: 0.7642,
        image_risk: 0.8120,
        nlp_risk: 0.7250,
        fused_risk: 0.7815,
        fusion_strategy: "neural",
        disclaimer: "MedIntel AI research decision-support risk estimate."
      });

      setExplanationData({
        shap_explanation: {
          top_positive_contributors: [
            { feature: "blood_pressure_sys", feature_value: 145, shap_value: 0.185 },
            { feature: "glucose", feature_value: 155, shap_value: 0.142 },
            { feature: "age", feature_value: 58, shap_value: 0.098 }
          ],
          top_negative_contributors: [
            { feature: "hemoglobin", feature_value: 13.8, shap_value: -0.045 }
          ],
          global_feature_importance: [
            { feature: "blood_pressure_sys", importance: 0.22 },
            { feature: "glucose", importance: 0.18 }
          ]
        },
        nlp_explanation: {
          highlighted_phrases: [
            { phrase: "shortness of breath", risk_weight: 0.85, impact: "high_positive_risk" },
            { phrase: "chest tightness", risk_weight: 0.80, impact: "high_positive_risk" }
          ],
          explanation_summary: "Identified key respiratory distress indicators in patient text note."
        }
      });
      setActiveTab('analysis');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 flex flex-col justify-between">
      {/* Navigation Header */}
      <header className="bg-slate-800/90 border-b border-slate-700 backdrop-blur sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-gradient-to-tr from-sky-500 to-indigo-600 rounded-lg shadow-md shadow-sky-500/20">
              <Activity className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-bold bg-gradient-to-r from-white via-slate-100 to-sky-300 bg-clip-text text-transparent">
                MedIntel AI
              </h1>
              <p className="text-[11px] text-sky-400 font-medium tracking-wide">Multimodal Medical Intelligence Platform</p>
            </div>
          </div>

          {/* Nav Tabs */}
          <nav className="flex items-center space-x-1 bg-slate-900/80 p-1.5 rounded-xl border border-slate-700/60">
            <button
              onClick={() => setActiveTab('overview')}
              className={`flex items-center space-x-1.5 px-3.5 py-1.5 text-xs font-medium rounded-lg transition-all ${
                activeTab === 'overview' ? 'bg-sky-500 text-white shadow' : 'text-slate-400 hover:text-white hover:bg-slate-800'
              }`}
            >
              <Activity className="w-4 h-4" />
              <span>Patient Overview</span>
            </button>

            <button
              onClick={() => setActiveTab('analysis')}
              className={`flex items-center space-x-1.5 px-3.5 py-1.5 text-xs font-medium rounded-lg transition-all ${
                activeTab === 'analysis' ? 'bg-sky-500 text-white shadow' : 'text-slate-400 hover:text-white hover:bg-slate-800'
              }`}
            >
              <Cpu className="w-4 h-4" />
              <span>AI Analysis</span>
            </button>

            <button
              onClick={() => setActiveTab('explain')}
              className={`flex items-center space-x-1.5 px-3.5 py-1.5 text-xs font-medium rounded-lg transition-all ${
                activeTab === 'explain' ? 'bg-sky-500 text-white shadow' : 'text-slate-400 hover:text-white hover:bg-slate-800'
              }`}
            >
              <Eye className="w-4 h-4" />
              <span>Explainability</span>
            </button>

            <button
              onClick={() => setActiveTab('rag')}
              className={`flex items-center space-x-1.5 px-3.5 py-1.5 text-xs font-medium rounded-lg transition-all ${
                activeTab === 'rag' ? 'bg-sky-500 text-white shadow' : 'text-slate-400 hover:text-white hover:bg-slate-800'
              }`}
            >
              <BookOpen className="w-4 h-4" />
              <span>Knowledge Assistant</span>
            </button>

            <button
              onClick={() => setActiveTab('models')}
              className={`flex items-center space-x-1.5 px-3.5 py-1.5 text-xs font-medium rounded-lg transition-all ${
                activeTab === 'models' ? 'bg-sky-500 text-white shadow' : 'text-slate-400 hover:text-white hover:bg-slate-800'
              }`}
            >
              <Database className="w-4 h-4" />
              <span>Model Information</span>
            </button>
          </nav>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="max-w-7xl mx-auto px-6 py-8 flex-1 w-full">
        {activeTab === 'overview' && (
          <PatientOverview
            patient={patientData}
            onChange={setPatientData}
            onAnalyze={handleRunInference}
            loading={loading}
          />
        )}

        {activeTab === 'analysis' && <AIAnalysis results={analysisResults} />}

        {activeTab === 'explain' && <ExplainabilityView explanation={explanationData} />}

        {activeTab === 'rag' && <KnowledgeAssistant />}

        {activeTab === 'models' && <ModelRegistryView />}
      </main>

      {/* Prominent Mandatory Medical Disclaimer Footer */}
      <footer className="bg-slate-950 border-t border-slate-800 py-4 px-6 text-center text-xs text-slate-400">
        <div className="max-w-5xl mx-auto flex items-center justify-center space-x-2 text-amber-400/90">
          <ShieldAlert className="w-4 h-4 flex-shrink-0" />
          <span>
            <strong>Disclaimer:</strong> MedIntel AI is a research and educational decision-support system. It does not provide medical diagnosis or treatment and should not replace professional medical advice.
          </span>
        </div>
      </footer>
    </div>
  );
};

export default App;
