import React from 'react';
import FloatingChatbot from '@site/src/components/FloatingChatbot';

export default function Root({children}: {children: React.ReactNode}): JSX.Element {
  return (
    <>
      {children}
      <FloatingChatbot />
    </>
  );
}
