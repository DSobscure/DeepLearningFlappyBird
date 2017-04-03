using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DataParser
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Input file name");
            string fileName = Console.ReadLine();
            string[] lines = File.ReadAllLines(fileName);
            Console.WriteLine($"{lines.Length} lines");

            using (TextWriter logger = File.CreateText("Records"))
            {
                for (int i = 0; i < lines.Length - 199; i+=200)
                {
                    float sum = 0;
                    for (int j = 0; j < 200; j++)
                    {
                        sum += float.Parse(lines[i + j]);
                    }
                    logger.WriteLine("{0}", sum / 200);
                }
            }

            Console.WriteLine("End");
            Console.ReadLine();
        }
    }
}
