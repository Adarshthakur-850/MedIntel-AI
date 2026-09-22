import React from 'react';
import { User, Activity, Heart, Thermometer, Droplet, FileText, AlertCircle } from 'lucide-react';

export interface PatientData {
  patient_id: string;
  age: number;
  sex: string;
  blood_pressure_sys: number;
  blood_pressure_dia: number;
  heart_rate: number;
  temperature: number;
  glucose: number;
  cholesterol: number;
  hemoglobin: number;
  creatinine: number;
  bmi: number;
  smoking_status: string;
  diabetes_history: number;
  clinical_note: string;
}

interface Props {
  patient: PatientData;
  onChange: (updated: PatientData) => void;
  onAnalyze: () => void;
  loading: boolean;
}

export const PatientOverview: React.FC<Props> = ({ patient, onChange, onAnalyze, loading }) => {
  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) => {
    const { name, value, type } = e.target;
    let parsedVal: any = value;
    if (type === 'number') {
      parsedVal = parseFloat(value) || 0;
    }
    onChange({ ...patient, [name]: parsedVal });
  };

  return (
    <div className="bg-slate-800 border border-slate-700 rounded-xl p-6 shadow-xl">
      <div className="flex items-center justify-between mb-6 pb-4 border-b border-slate-700">
        <div className="flex items-center space-x-3">
          <User className="w-6 h-6 text-sky-400" />
          <h2 className="text-xl font-semibold text-white">Patient Clinical Data</h2>
        </div>
        <span className="px-3 py-1 bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs font-medium rounded-full flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
          Input Validated
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        <div>
          <label className="block text-xs font-medium text-slate-400 mb-1">Patient ID</label>
          <input
            type="text"
            name="patient_id"
            value={patient.patient_id}
            onChange={handleChange}
            className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-sky-500"
          />
        </div>

        <div>
          <label className="block text-xs font-medium text-slate-400 mb-1">Age (Years)</label>
          <input
            type="number"
            name="age"
            value={patient.age}
            onChange={handleChange}
            className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-sky-500"
          />
        </div>

        <div>
          <label className="block text-xs font-medium text-slate-400 mb-1">Sex</label>
          <select
            name="sex"
            value={patient.sex}
            onChange={handleChange}
            className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-sky-500"
          >
            <option value="M">Male (M)</option>
            <option value="F">Female (F)</option>
          </select>
        </div>

        <div>
          <label className="block text-xs font-medium text-slate-400 mb-1">Systolic BP (mmHg)</label>
          <input
            type="number"
            name="blood_pressure_sys"
            value={patient.blood_pressure_sys}
            onChange={handleChange}
            className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-sky-500"
          />
        </div>

        <div>
          <label className="block text-xs font-medium text-slate-400 mb-1">Diastolic BP (mmHg)</label>
          <input
            type="number"
            name="blood_pressure_dia"
            value={patient.blood_pressure_dia}
            onChange={handleChange}
            className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-sky-500"
          />
        </div>

        <div>
          <label className="block text-xs font-medium text-slate-400 mb-1">Heart Rate (bpm)</label>
          <input
            type="number"
            name="heart_rate"
            value={patient.heart_rate}
            onChange={handleChange}
            className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-sky-500"
          />
        </div>

        <div>
          <label className="block text-xs font-medium text-slate-400 mb-1">Glucose (mg/dL)</label>
          <input
            type="number"
            name="glucose"
            value={patient.glucose}
            onChange={handleChange}
            className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-sky-500"
          />
        </div>

        <div>
          <label className="block text-xs font-medium text-slate-400 mb-1">Cholesterol (mg/dL)</label>
          <input
            type="number"
            name="cholesterol"
            value={patient.cholesterol}
            onChange={handleChange}
            className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-sky-500"
          />
        </div>

        <div>
          <label className="block text-xs font-medium text-slate-400 mb-1">Creatinine (mg/dL)</label>
          <input
            type="number"
            name="creatinine"
            step="0.1"
            value={patient.creatinine}
            onChange={handleChange}
            className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-sky-500"
          />
        </div>
      </div>

      <div className="mt-5">
        <label className="block text-xs font-medium text-slate-400 mb-1">Clinical Symptoms / Progress Note</label>
        <textarea
          name="clinical_note"
          rows={3}
          value={patient.clinical_note}
          onChange={handleChange}
          className="w-full bg-slate-900 border border-slate-700 rounded-lg p-3 text-sm text-white focus:outline-none focus:border-sky-500 resize-none"
        />
      </div>

      <div className="mt-6 flex justify-end">
        <button
          onClick={onAnalyze}
          disabled={loading}
          className="px-6 py-2.5 bg-gradient-to-r from-sky-500 to-indigo-600 hover:from-sky-400 hover:to-indigo-500 text-white font-medium rounded-lg shadow-lg shadow-sky-500/20 transition-all duration-200 disabled:opacity-50"
        >
          {loading ? "Processing Inference..." : "Run Multimodal AI Inference"}
        </button>
      </div>
    </div>
  );
};
