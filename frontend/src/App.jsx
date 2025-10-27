import React, { useState } from 'react';
//import logo from '../public/logo.svg';

export default function App() {
  const [query, setQuery] = useState('');
  const [submitted, setSubmitted] = useState(false);

  const answerData = {
    verses: [
      { index: '1.1.0.121', text: 'This is the text of the first relevant verse.' },
      { index: '1.1.0.122', text: 'Another verse text, not displayed in table.' }
    ],
    summary: 'This is a summary of the answer.',
    entities: ['Arjuna', 'Krishna', 'Dharma']
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (query.trim() !== '') setSubmitted(true);
  };

  const handleRegenerate = () => setSubmitted(false);

  const featureList = [
    'Placeholder 1: Answer to your dharma queries',
    'Placeholder 2: the only RAGging we allow on campus',
    'Placeholder 3: The real Mahabharat is being fought for our grades in the GenAI classroom'
  ];

  return (
    <div className="flex h-screen">
      <div className="w-64 bg-beige-bg p-6 flex flex-col gap-4">
        <h2 className="text-xl font-bold rust">Menu</h2>
        <button className="text-left p-2 hover:bg-rust/10 rounded">New Query</button>
        <button className="text-left p-2 hover:bg-rust/10 rounded">Existing Queries</button>
        <button className="text-left p-2 hover:bg-rust/10 rounded">About Mahānveṣa</button>
      </div>

      <div className="flex-1 p-8">
        <form onSubmit={handleSubmit}>
          <input
            type="text"
            placeholder="Type your Mahābharat query here"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="w-full p-3 border border-gray-300 rounded mb-6 focus:outline-none focus:ring-2 focus:ring-rust/50"
          />
        </form>

        {!submitted ? (
          <div className="text-center mt-20">
            {/*<img src={logo} alt="Mahānveṣa Logo" className="mx-auto mb-4 w-20 h-20" /> */}
            <h1 className="text-3xl font-bold rust mb-6">
              Mahānveṣa: an AI tool for your Mahābharata Queries
            </h1>
            <div className="space-y-2">
              {featureList.map((feature, idx) => (
                <p key={idx} className="text-gray-700">{feature}</p>
              ))}
            </div>
          </div>
        ) : (
          <div>
            <div className="flex items-center mb-6">
              {/*<img src={logo} alt="logo" className="w-8 h-8 mr-2" />*/}
              <span className="rust font-semibold text-lg">Mahānveṣa</span>
            </div>

            <table className="w-full border border-gray-300">
              <thead className="bg-beige-bg">
                <tr>
                  <th className="border px-4 py-2">Retrieved Verses</th>
                  <th className="border px-4 py-2">Answer Summary</th>
                  <th className="border px-4 py-2">Related Entities</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td className="border px-4 py-2">
                    {answerData.verses.map((v, idx) => (
                      <div key={idx}>
                        <a
                          href={`https://example.com/verse/${v.index}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-rust underline"
                        >
                          {v.index}
                        </a>
                      </div>
                    ))}
                    <div className="mt-2 text-gray-700">{answerData.verses[0].text}</div>
                  </td>
                  <td className="border px-4 py-2">{answerData.summary}</td>
                  <td className="border px-4 py-2">{answerData.entities.join(', ')}</td>
                </tr>
              </tbody>
            </table>

            <button
              onClick={handleRegenerate}
              className="mt-4 px-4 py-2 bg-rust text-white rounded hover:bg-rust/90"
            >
              Edit / Regenerate
            </button>
          </div>
        )}
      </div>
    </div>
  );
}