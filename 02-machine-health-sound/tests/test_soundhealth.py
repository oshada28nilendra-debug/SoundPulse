import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy.io import wavfile
from soundhealth.core import Config,SoundMonitor,read_audio,features,mel_filterbank
from soundhealth.data import synthetic,read_manifest
from soundhealth.evaluate import metrics,perturb


class SoundHealthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary=tempfile.TemporaryDirectory()
        cls.root=Path(cls.temporary.name)
        cls.manifest=synthetic(cls.root/'data')
        _,cls.train,cls.cal,cls.test=read_manifest(cls.manifest)
        cls.model=SoundMonitor().fit(cls.train[:12]).calibrate(cls.cal)

    @classmethod
    def tearDownClass(cls):cls.temporary.cleanup()

    def test_detects_synthetic_defect_signal(self):
        normal=[self.model.predict(read_audio(r['audio']))['score'] for r in self.test[:3]]
        abnormal=[self.model.predict(read_audio(r['audio']))['score'] for r in self.test[20:23]]
        self.assertGreater(float(np.median(abnormal)),float(np.median(normal)))

    def test_resamples_integer_stereo_without_amplitude_error(self):
        sr=8000;t=np.arange(sr*2)/sr
        mono=.25*np.sin(2*np.pi*400*t)
        path=self.root/'stereo.wav'
        wavfile.write(path,sr,np.int16(np.stack([mono,mono],axis=1)*32767))
        wave=read_audio(path)
        self.assertEqual(len(wave),32000)
        self.assertAlmostEqual(float(np.sqrt(np.mean(wave**2))),.25/np.sqrt(2),places=3)
        frequency=np.fft.rfftfreq(len(wave),1/16000)[np.abs(np.fft.rfft(wave)).argmax()]
        self.assertAlmostEqual(frequency,400,delta=1)

    def test_gain_normalized_features_are_invariant(self):
        wave=read_audio(self.train[0]);config=Config(level_normalize=True)
        a=features(wave,config)[0];b=features(wave*.5,config)[0]
        np.testing.assert_allclose(a,b,atol=2e-4)

    def test_noise_level_and_determinism(self):
        wave=read_audio(self.train[0])
        a=perturb(wave,'added_white_noise_10dB','clip-0',17)
        b=perturb(wave,'added_white_noise_10dB','clip-0',17)
        np.testing.assert_array_equal(a,b)
        snr=10*np.log10(np.mean(wave.astype(float)**2)/np.mean((a-wave).astype(float)**2))
        self.assertAlmostEqual(float(snr),10,places=4)

    def test_serialization_preserves_score_and_trace(self):
        path=self.root/'model.npz';self.model.save(path)
        loaded=SoundMonitor.load(path)
        wave=read_audio(self.test[20]['audio'])
        a,b=self.model.predict(wave),loaded.predict(wave)
        self.assertAlmostEqual(a['score'],b['score'],places=5)
        np.testing.assert_allclose(a['window_scores'],b['window_scores'],atol=1e-5)
        self.assertEqual(a['anomalous'],b['anomalous'])

    def test_calibration_and_training_overlap_rejected(self):
        with self.assertRaisesRegex(ValueError,'disjoint'):
            self.model.calibrate(self.train[:10])

    def test_duplicate_content_cannot_cross_manifest_splits(self):
        payload=json.loads(self.manifest.read_text())
        payload['calibration'][0]=payload['train'][0]
        bad=self.manifest.with_name('leaky.json');bad.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError,'leakage'):
            read_manifest(bad)

    def test_silence_and_short_audio_are_not_healthy_predictions(self):
        with self.assertRaisesRegex(ValueError,'insufficient signal'):
            self.model.predict(np.zeros(16000,dtype=np.float32))
        with self.assertRaisesRegex(ValueError,'one second'):
            self.model.predict(np.ones(500,dtype=np.float32)*.1)

    def test_nonfinite_audio_rejected(self):
        with self.assertRaises(ValueError):
            self.model.predict(np.full(16000,np.nan))

    def test_threshold_is_normal_only_order_statistic(self):
        scores=sorted(self.model.predict(read_audio(p))['score'] for p in self.cal)
        # ceil(21*.9)=19 => zero-based index18
        self.assertAlmostEqual(self.model.threshold,scores[18],places=5)
        with self.assertRaisesRegex(ValueError,'not enough'):
            self.model.calibrate(self.cal[:2],alpha=.01)

    def test_metrics_keep_false_alarms_separate_from_recall(self):
        rows=[{'label':y,'score':s} for y,s in zip([0,0,1,1],[.1,.8,.7,.9])]
        result=metrics(rows,.75)
        self.assertEqual(result['false_alarms'],1)
        self.assertEqual(result['detected_anomalies'],1)
        self.assertEqual(result['false_alarm_rate'],.5)

    def test_feature_trace_dimensions_and_mel_support(self):
        wave=read_audio(self.train[0]);config=Config()
        matrix,times,spec=features(wave,config)
        self.assertEqual(matrix.shape[1],256)
        self.assertEqual(len(matrix),len(times))
        self.assertTrue(np.all(np.diff(times)>0))
        self.assertLessEqual(times[-1],len(wave)/16000)
        self.assertEqual(spec.shape[0],64)
        bank=mel_filterbank(config)
        np.testing.assert_allclose(bank.sum(axis=1),1,atol=1e-6)


if __name__=='__main__':unittest.main()
