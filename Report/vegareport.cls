\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{vegareport}[2022/08/16 Vega Report Class (Blue)]
\LoadClass[a4paper]{report}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% F.A.Q.
% Options: > russian: pass if document is in Russian to change the language of pre-defined text
%          > 
%          > 
%          > 
%          > 
%          > 
%          > 
%          > 
%          > 
%          > 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Packages and Options
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{cmap}
\RequirePackage[T1,T2A]{fontenc}
\RequirePackage[utf8]{inputenc}
\RequirePackage{kvoptions}

\RequirePackage{xcolor}
\RequirePackage{tikz}
\RequirePackage{geometry}
\RequirePackage{fancyhdr}
\RequirePackage{graphicx}

\RequirePackage{booktabs}
\RequirePackage{bbm}
\RequirePackage{hyperref}

\RequirePackage{amsmath,amssymb}
\RequirePackage{mathtext}
\RequirePackage{amsthm}

\RequirePackage[backend=biber,style=alphabetic,sorting=ynt]{biblatex}
\RequirePackage{csquotes}

\DeclareBoolOption{russian}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Headers and Footers
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\pagestyle{fancy}
\setlength{\headheight}{14pt}
\rhead{\includegraphics[width=0.08\linewidth]{vega_logo_smaller.png}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Colors, Fonts, and General Settings
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{paratype}

\definecolor{color1bg}{HTML}{2C3389}
\definecolor{color2bg}{HTML}{EFA69D}
\definecolor{color3bg}{HTML}{B16D95}

\definecolor{colormainblue}{HTML}{2B3189}

\newcommand{\headlinecolor}{\normalcolor}

\hypersetup
{
	colorlinks=true,
	linkcolor=blue,
	filecolor=magenta,      
	urlcolor=cyan,
	citecolor=gray,
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Language-dependent commands
% Default:   English
% Available: English, Russian
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newcommand{\textDocumentType}   {Student Research Group Report}

\newcommand{\textTableofcontents}{Table of Contents}
\newcommand{\textListoffigures}  {List of Figures}
\newcommand{\textListoftables}   {List of Tables}
\newcommand{\textIntroduction}   {Introduction}
\newcommand{\textConclusion}     {Conclusion}
\newcommand{\textReferences}     {Bibliography}

\newcommand{\textDefinition}     {Definition}
\newcommand{\textTheorem}        {Theorem}
\newcommand{\textLemma}          {Lemma}
\newcommand{\textCorollary}      {Corollary}
\newcommand{\textExample}        {Example}
\newcommand{\textProposition}    {Proposition}
\newcommand{\textRemark}         {Remark}


\ProcessKeyvalOptions*
\ifvegareport@russian
    \renewcommand{\textDocumentType}   {Отчет по студенческой научной группе}
    \renewcommand{\textTableofcontents}{Содержание}
    \renewcommand{\textListoffigures}  {Список иллюстраций}
    \renewcommand{\textListoftables}   {Список таблиц}
    \renewcommand{\textIntroduction}   {Введение}
    \renewcommand{\textConclusion}     {Заключение}
    \renewcommand{\textReferences}     {Список литературы}
    \renewcommand{\textDefinition}     {Определение}
    \renewcommand{\textTheorem}        {Теорема}
    \renewcommand{\textLemma}          {Лемма}
    \renewcommand{\textCorollary}      {Следствие}
    \renewcommand{\textExample}        {Пример}
    \renewcommand{\textProposition}    {Предложение}
    \renewcommand{\textRemark}         {Замечание}
\fi

\ProcessKeyvalOptions*
\ifvegareport@russian
    \RequirePackage[russian]{babel}
\else
    \RequirePackage[english]{babel}
\fi

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Title page etc
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\let\TOC\tableofcontents
\let\LOT\listoftables
\let\LOF\listoffigures
\renewcommand{\tableofcontents}{\newpage\phantomsection\addcontentsline{toc}{chapter}{\textTableofcontents}\TOC}
\renewcommand{\listoffigures}  {\newpage\phantomsection\addcontentsline{toc}{chapter}{\textListoffigures}\LOF}
\renewcommand{\listoftables}   {\newpage\phantomsection\addcontentsline{toc}{chapter}{\textListoftables}\LOT}

\renewcommand{\maketitle}
{
    \pagestyle{empty}
    \begin{titlepage}\newgeometry{left=2cm,right=3.125cm,bottom=1cm}
        \begin{tikzpicture}[remember picture,overlay]
            \node [fill, rectangle, top color=color1bg,  bottom color=color1bg, anchor=north, minimum width=\paperwidth, minimum height=\paperheight,shading angle=30] (box) at (current page.north){};
            \node [anchor=north east] at (current page.north east){\includegraphics[width=6.25cm]{vega_letter_white.pdf}};
        \end{tikzpicture}

        {
            \color{white}
            \vspace{4.75cm}
            \begin{large}
                \noindent\textDocumentType
            \end{large}
            \vspace{0.25cm}

            \begin{LARGE}
                \noindent{\textbf{\@title}}
            \end{LARGE}

            \noindent\hrulefill

            \vspace{0.25cm}

            \begin{Large}
            \noindent{\@author}
            \end{Large}
        }
        \end{titlepage}\pagestyle{fancy}
        \newpage
        \restoregeometry
        \pagenumbering{roman}
        \tableofcontents
        \listoffigures
        \listoftables
        \newpage
        \pagenumbering{arabic}\setcounter{page}{3}
}


\newcommand{\introduction}{\newpage\chapter*{\textIntroduction}\addcontentsline{toc}{part}{\textIntroduction}}
\newcommand{\conclusion}  {\newpage\chapter*{\textConclusion}\addcontentsline{toc}{part}{\textConclusion}}
\newcommand{\references}  {\newpage\phantomsection\addcontentsline{toc}{part}{\textReferences}\printbibliography}

\newtheorem{definition}{\textDefinition}
\newtheorem{theorem}{\textTheorem}
\newtheorem{corollary}{\textCorollary}[theorem]
\newtheorem{lemma}[theorem]{\textLemma}
\newtheorem{example}{\textExample}
\newtheorem{proposition}{\textProposition}
\newtheorem*{remark}{\textRemark}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Maths commands
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\DeclareMathOperator*{\sgn}{sgn}
\DeclareMathOperator*{\var}{var}   
\DeclareMathOperator*{\cov}{cov}
\DeclareMathOperator*{\law}{Law}

\newcommand{\1}{\mathbbm{1}} 
\newcommand{\R}{\mathbb{R}}
\newcommand{\N}{\mathbb{N}}
\renewcommand{\P}{\mathbb{P}}
\newcommand{\E}{\mathbb{E}}

\newcommand{\independent}{\perp\!\!\!\!\perp}

