import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Layout } from './components/Layout';
import { LandingPage } from './pages/LandingPage';
import { LibraryPage } from './pages/LibraryPage';
import { CreatePage } from './pages/CreatePage';
import { StudioPage } from './pages/StudioPage';
import { ReviewPage } from './pages/ReviewPage';
import { DashboardPage } from './pages/DashboardPage';
import { TutorialPage } from './pages/TutorialPage';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<LandingPage />} />
          <Route path="library" element={<LibraryPage />} />
          <Route path="create" element={<CreatePage />} />
          <Route path="studio" element={<StudioPage />} />
          <Route path="review" element={<ReviewPage />} />
          <Route path="dashboard" element={<DashboardPage />} />
          <Route path="tutorial" element={<TutorialPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
