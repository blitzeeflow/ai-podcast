from transformers import pipeline

summarizer = pipeline("summarization", model="Falconsai/text_summarization")

ARTICLE = """ 
FCSM, which stands for “Fuel Cell System Manufacturing,” was established in 2017 as a joint venture between GM and Honda. The two automakers have also collaborated on battery electric vehicles, including the Honda Prologue, Acura ZDX, and Cruise Origin.

FCSM’s 70,000-square-foot facility in Brownstown, Michigan, was built with an $83 million joint investment by GM and Honda. The companies call it “the first large-scale manufacturing joint venture to build fuel cells.”

Hydrogen has found little success in the passenger car market. Honda was one of the only companies to sell a hydrogen-powered car — the Clarity — before it was discontinued in 2017. The problem stems from the near-total absence of a refueling infrastructure. Automakers are now pivoting to work trucks and construction equipment, theorizing that it will be easier to build hydrogen fueling stations for vehicles that operate in confined areas.

Hydrogen’s energy content by volume is low, which makes storing hydrogen a challenge because it requires high pressures, low temperatures, or chemical processes to be stored compactly. Overcoming this challenge is important for light-duty vehicles because they often have limited size and weight capacity for fuel storage.
"""
print(summarizer(ARTICLE, max_length=230, min_length=30, do_sample=False)[0])
